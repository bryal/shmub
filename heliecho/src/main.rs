#![feature(slice_patterns)]

extern crate chfft;
extern crate cpal;
extern crate num_complex;
extern crate palette;
extern crate serial;
extern crate shmub_common;

use serial::prelude::*;
use std::{io, thread};
use std::io::prelude::*;
use std::sync::mpsc;
use std::time;
use chfft::RFft1D as Fft;
use cpal::{Format, SampleFormat, SampleRate, StreamData, UnknownTypeInputBuffer};
use shmub_common::*;
use palette::Hsv;
use palette::rgb::{Rgb, Srgb};

type Rgb8 = Srgb<u8>;

const SAMPLE_RATE: u32 = 48000;
// Must be power of 2
const FRAMES_PER_PERIOD: usize = 1024;
const MAX_DB_DECAY_PER_PERIOD: f32 = 1.0 - 0.1 * FRAMES_PER_PERIOD as f32 / SAMPLE_RATE as f32;
const MAX_FREQ: f32 = 2800.0;
const ADALIGHT_BAUDRATE: serial::BaudRate = serial::Baud115200;
const N_LEDS: usize = 49;
const BLACK_LEVEL: f32 = 0.15;

fn main() {
    println!("Heliecho");
    println!(
        "Convert audio stream to pretty colors based on the properties of the sound, \
         and stream as LED data to FastLED compatible serial device."
    );

    let led_device = parse_args();
    let led_tx = init_write_thread(led_device);

    let audio_device = prompt_audio_device();
    let samples_rx = init_audio_thread(audio_device);

    audio_to_leds_loop(samples_rx, led_tx);
}

fn audio_to_leds_loop(samples_rx: mpsc::Receiver<Vec<f32>>, led_tx: mpsc::SyncSender<Rgb8>) {
    let mut buf = [[0f32; N_CHANNELS]; FRAMES_PER_PERIOD];
    let mut buf_i = 0usize;
    let mut prev_print_t = time::Instant::now();
    let mut max_db = 1.0;
    println!("");
    loop {
        let samples = samples_rx.recv().expect("error receiving on channel");
        let frames = samples.chunks(N_CHANNELS).map(|w| [w[0], w[1]]);
        for frame in frames {
            buf[buf_i] = frame;
            buf_i += 1;
            if buf_i == FRAMES_PER_PERIOD {
                write_frames_as_colors(&buf, &led_tx, &mut max_db, &mut prev_print_t);
                max_db *= MAX_DB_DECAY_PER_PERIOD;
                if max_db < 1.0 {
                    max_db = 1.0;
                }
                buf_i = 0;
            }
        }
    }
}

/// Initialize a thread for serial writing given a serial port, baud rate, header to write before
/// each data write, and buffer with the actual led color data.
fn init_write_thread(mut serial_con: serial::SystemPort) -> mpsc::SyncSender<Rgb8> {
    use std::io::Write;
    let (tx, rx) = mpsc::sync_channel::<Rgb8>(0);
    thread::spawn(move || {
        let count_high = ((N_LEDS - 1) >> 8) as u8; // LED count high byte
        let count_low = ((N_LEDS - 1) & 0xff) as u8; // LED count low byte
        let mut color_buf = [0; 6 + 3 * N_LEDS];
        // Header
        color_buf[0] = 'A' as u8;
        color_buf[1] = 'd' as u8;
        color_buf[2] = 'a' as u8;
        color_buf[3] = count_high;
        color_buf[4] = count_low;
        color_buf[5] = count_high ^ count_low ^ 0x55; // Checksum
        let mut prev_color = Rgb::new(0, 0, 0);
        loop {
            let recv_color = rx.try_recv().unwrap_or(prev_color);
            let color = smooth_color(prev_color, recv_color);
            for n in 0..N_LEDS {
                color_buf[6 + 3 * n] = color.red;
                color_buf[6 + 3 * n + 1] = color.green;
                color_buf[6 + 3 * n + 2] = color.blue;
            }
            match serial_con.write(&color_buf[..]) {
                Ok(bn) if bn == color_buf.len() => (),
                Ok(_) => println!("Failed to write all bytes of RGB data"),
                Err(e) => println!("Failed to write RGB data, {}", e),
            }
            prev_color = color;
        }
    });
    tx
}

fn init_audio_thread(audio_device: cpal::Device) -> mpsc::Receiver<Vec<f32>> {
    let format = Format {
        channels: N_CHANNELS as u16,
        sample_rate: SampleRate(SAMPLE_RATE),
        data_type: SampleFormat::F32,
    };
    let event_loop = cpal::EventLoop::new();
    let stream_id = event_loop
        .build_input_stream(&audio_device, &format)
        .expect(&format!(
            "Failed to build input stream for device {} with format {:?}",
            audio_device.name(),
            format
        ));
    event_loop.play_stream(stream_id);
    let (samples_tx, samples_rx) = mpsc::sync_channel(4);
    thread::spawn(move || {
        event_loop.run(move |_, data| {
            let samples = to_f32_buffer(as_input_buffer(&data));
            match samples_tx.try_send(samples) {
                Err(mpsc::TrySendError::Disconnected(_)) => panic!("channel disconnected"),
                _ => (),
            }
        })
    });
    samples_rx
}

fn write_frames_as_colors(
    stereo_data: &[[f32; 2]; FRAMES_PER_PERIOD],
    led_tx: &mpsc::SyncSender<Rgb8>,
    max_db: &mut f32,
    prev_print_t: &mut time::Instant,
) {
    let mut bin_amps_db = stereo_pcm_to_db_bins(stereo_data);
    apply_amplification(&mut bin_amps_db);
    let (freq, amp) = max_amp(&bin_amps_db);
    let level = norm_db(amp, max_db);
    let x = ((freq - 20.0).max(0.0) / MAX_FREQ - 1.0).min(0.0);
    let hue = (1.0 - x.powi(2)) * 240.0;
    let sat = 1.0;
    let mut val = level.powf(4.0);
    if val < BLACK_LEVEL {
        val = 0.0;
    }
    let hsv = Hsv::new(hue, sat, val);
    let rgb = Rgb::from(hsv);
    let rgb8 = rgb.into_format::<u8>();
    led_tx
        .send(rgb8)
        .expect("Error sending RGB value over channel");

    let dt = prev_print_t.elapsed();
    if dt > time::Duration::from_millis(100) {
        *prev_print_t = time::Instant::now();
        print!("\rf: {:6.0}, db: {:4.1}, vol: {:1.3}", freq, amp, level,);
        io::stdout().flush().expect("Error flushing stdout");
    }
}

/// bin == index of fft where there is a corresponding frequence
fn stereo_pcm_to_db_bins(
    stereo_data: &[[f32; 2]; FRAMES_PER_PERIOD],
) -> [f32; (FRAMES_PER_PERIOD >> 1) + 1] {
    let mut avg_data = [0.0; FRAMES_PER_PERIOD];
    for (i, &[l, r]) in stereo_data.iter().enumerate() {
        avg_data[i] = (l + r) / 2.0;
    }
    let mut fft = Fft::new(FRAMES_PER_PERIOD);
    // X = fft(x)
    let bin_amps_complex = fft.forward(&avg_data[..]);
    let mut bin_amps_db = [0.0f32; (FRAMES_PER_PERIOD >> 1) + 1];
    for (i, &c) in bin_amps_complex.iter().enumerate() {
        let amp = (c.re.powi(2) + c.im.powi(2)).sqrt();
        bin_amps_db[i] = 20.0 * amp.log(10.0);
    }
    bin_amps_db
}

fn apply_amplification(amps: &mut [f32]) {
    for (i, a) in amps.iter_mut().enumerate() {
        let f = bin_to_freq(i as f32);
        let end = 1.2;
        *a *= 1.0 + (end * (f / MAX_FREQ).min(1.0));
    }
}

fn bin_to_freq(i: f32) -> f32 {
    (i * SAMPLE_RATE as f32) / FRAMES_PER_PERIOD as f32
}

/// Normalize a decibel value to [0, 1]
fn norm_db(db: f32, max_db: &mut f32) -> f32 {
    if db > *max_db {
        *max_db = db;
    }
    let x = db / *max_db;
    if x > 1.0 {
        1.0
    } else {
        (1.0 - x) * x + x * (1.0 / (1.0 + (-10.0 * (x - 0.5)).exp()))
    }
}

fn max_amp(amps: &[f32]) -> (f32, f32) {
    let mut max_bin = 0;
    let mut max_amp = 0.0;
    for (i, &a) in amps.iter().enumerate() {
        if a > max_amp {
            max_bin = i;
            max_amp = a;
        }
    }
    let mut snd_bin = 0;
    let mut snd_amp = 0.0;
    for (i, &a) in amps.iter().enumerate() {
        if i != max_bin && a > snd_amp {
            snd_bin = i;
            snd_amp = a;
        }
    }
    let bin = if snd_bin == max_bin + 1 || (max_bin > 0 && snd_bin == max_bin - 1) {
        (snd_bin as f32 * snd_amp + max_bin as f32 * max_amp) / (snd_amp + max_amp)
    } else {
        max_bin as f32
    };
    (bin_to_freq(bin), max_amp)
}

/// Go faster towards light, slower towards dark
fn smooth_color(from: Rgb8, to: Rgb8) -> Rgb8 {
    let from_hsv = Hsv::from(from.into_format::<f32>());
    let to_hsv = Hsv::from(to.into_format::<f32>());
    let from_hue = from_hsv.hue.to_positive_degrees();
    let to_hue = to_hsv.hue.to_positive_degrees();
    let hue_diff = to_hue - from_hue;
    let hue = from_hue + 0.04 * hue_diff;
    let sat = 1.0;
    let from_val = from_hsv.value;
    let val_diff = to_hsv.value - from_val;
    let val = from_val + 0.2 * val_diff;
    Rgb::from(Hsv::new(hue, sat, val)).into_format::<u8>()
}

fn to_f32_buffer(unknown_buf: &UnknownTypeInputBuffer) -> Vec<f32> {
    match unknown_buf {
        UnknownTypeInputBuffer::U16(buf) => buf.iter().map(cpal::Sample::to_f32).collect(),
        UnknownTypeInputBuffer::I16(buf) => buf.iter().map(cpal::Sample::to_f32).collect(),
        UnknownTypeInputBuffer::F32(buf) => buf.to_vec(),
    }
}

fn as_input_buffer<'b>(data: &'b StreamData) -> &'b UnknownTypeInputBuffer<'b> {
    match data {
        StreamData::Input { buffer } => buffer,
        _ => panic!("Expected input data buffer"),
    }
}

fn prompt_audio_device() -> cpal::Device {
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

fn parse_args() -> serial::SystemPort {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() > 1 {
        let port = &args[1];
        let mut serial_con = serial::open(port).expect("Error opening serial port");
        serial_con
            .reconfigure(&|cfg| cfg.set_baud_rate(ADALIGHT_BAUDRATE))
            .expect("Error configuring serial connection");
        if args.len() > 2 {
            panic!("Unexpected argument `{}`", args[3])
        } else {
            serial_con
        }
    } else {
        panic!("Missing argument serial port")
    }
}
