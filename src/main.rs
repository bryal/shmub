extern crate chfft;
extern crate cpal;
extern crate hound;
extern crate num_complex;

use std::io::{self, BufRead, Write};
use std::time;
use cpal::{StreamData, UnknownTypeInputBuffer};
use chfft::RFft1D as Fft;
use num_complex::Complex;

// Must be power of 2
const SAMPLES_PER_PERIOD: usize = 4096;

const COMPLEX_ZERO: Complex<f32> = Complex { re: 0.0, im: 0.0 };

macro_rules! print_flush {
    ($s: expr) => {
        print_flush!($s,);
    };
    ($format_str: expr, $($args: expr),*) => {
        print!($format_str, $($args),*);
        io::stdout().flush().unwrap();
    };
}

fn main() {
    println!("Shmub Audio Client");
    println!("Capture audio from an input device and stream to a Shmub server");
    println!("");

    let device = prompt_device();
    let format = device
        .default_input_format()
        .expect("Failed to get input format");
    println!(
        "Selected device:\n  {}\n  {} channels, {}hz, {:?}",
        device.name(),
        format.channels,
        format.sample_rate.0,
        format.data_type
    );

    let event_loop = cpal::EventLoop::new();
    let stream_id = event_loop
        .build_input_stream(&device, &format)
        .expect("Failed to build input stream");
    event_loop.play_stream(stream_id);

    let mono_sample_rate = format.sample_rate.0 * format.channels as u32;
    let start_time = time::Instant::now();
    let mut buffer = [0.0f32; SAMPLES_PER_PERIOD];
    let mut buffer_size = 0usize;
    let mut processed_mono_samples = Vec::new();
    // TODO: channel that streams one sample at a time
    event_loop.run(move |_, data| {
        if start_time.elapsed() > time::Duration::from_secs(6) {
            after_event_loop(&processed_mono_samples, mono_sample_rate)
        }
        let samples = to_f32_buffer(as_input_buffer(&data))
            .windows(format.channels as usize)
            .map(|w| w.iter().sum::<f32>() / format.channels as f32)
            .collect::<Vec<_>>();
        let mut i = 0;
        loop {
            if buffer_size == SAMPLES_PER_PERIOD {
                let processed = process_buffer(&buffer, mono_sample_rate);
                processed_mono_samples.extend_from_slice(&processed);
                buffer_size = 0;
            }
            if i == samples.len() {
                break;
            } else {
                buffer[buffer_size] = samples[i];
                buffer_size += 1;
                i += 1;
            }
        }
    });
}

fn after_event_loop(samples: &[f32], sample_rate: u32) -> ! {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("recording.wav", spec).unwrap();
    for &sample in samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();

    std::process::exit(0)
}

fn process_buffer(x: &[f32], sample_rate: u32) -> Vec<f32> {
    let mut fft = Fft::new(x.len());
    // X = fft(x)
    let xx = fft.forward(x);
    // w = freq(i)
    let freq = |i| i as f32 * sample_rate as f32 / SAMPLES_PER_PERIOD as f32;
    // Cutoff frequency
    // w_c = sigma_f
    let wc = 500.0;
    // Gaussian filter
    let gg = |w: f32| (-(w * w) / (2.0 * wc * wc)).exp();
    // Y(w) = X(w) * G(w)
    let yy = (0..xx.len())
        .map(|i| xx[i] * gg(freq(i)))
        .collect::<Vec<_>>();
    // y = ifft(Y)
    let y = fft.backward(&yy);
    y
}

fn to_f32_buffer(unknown_buf: &UnknownTypeInputBuffer) -> Vec<f32> {
    match unknown_buf {
        UnknownTypeInputBuffer::U16(buf) => buf.iter()
            .map(|&x| x as f32 / std::u16::MAX as f32)
            .collect(),
        UnknownTypeInputBuffer::I16(buf) => buf.iter()
            .map(|&x| (x as f32 - std::i16::MIN as f32) / std::u16::MAX as f32)
            .collect(),
        UnknownTypeInputBuffer::F32(buf) => buf.to_vec(),
    }
}

fn as_input_buffer<'b>(data: &'b StreamData) -> &'b UnknownTypeInputBuffer<'b> {
    match data {
        StreamData::Input { buffer } => buffer,
        _ => panic!("Expected input data buffer"),
    }
}

fn prompt_device() -> cpal::Device {
    println!("Select input device to listen to:");
    let mut devices = cpal::input_devices().collect::<Vec<_>>();
    for (i, device) in devices.iter().enumerate() {
        println!(" {}) {}", i, device.name());
    }
    let line = prompt_line();
    let i = line.parse::<usize>().expect(&format!(
        "Failed to parse input as device index: line = \"{}\"",
        line
    ));
    if i >= devices.len() {
        panic!("Undefined device index");
    } else {
        devices.remove(i)
    }
}

fn prompt_line() -> String {
    print_flush!("> ");
    read_line()
}

fn read_line() -> String {
    let stdin = io::stdin();
    let s = stdin
        .lock()
        .lines()
        .next()
        .unwrap()
        .expect("Failed to read line from stdin");
    s
}
