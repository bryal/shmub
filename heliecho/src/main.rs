#![feature(slice_patterns)]
#![feature(const_fn)]

extern crate chfft;
extern crate cpal;
extern crate num;
extern crate palette;
extern crate serial;
extern crate shmub_common;

use serial::prelude::*;
use std::{io, thread};
use std::io::prelude::*;
use std::sync::mpsc;
use std::time;
use std::ops::{Add, Div, Range};
use num::One;
use chfft::RFft1D as Fft;
use cpal::{Format, SampleFormat, SampleRate, StreamData, UnknownTypeInputBuffer};
use shmub_common::*;
use palette::Hsv;
use palette::rgb::{Rgb, Srgb};

type Rgb8 = Srgb<u8>;

const SAMPLE_RATE: u32 = 48000;
// Must be power of 2
const FRAMES_PER_PERIOD: usize = 1024;
const MAX_AMPS_DECAY_PER_PERIOD: f32 = 1.0 - 0.03 * FRAMES_PER_PERIOD as f32 / SAMPLE_RATE as f32;
const ADALIGHT_BAUDRATE: serial::BaudRate = serial::Baud115200;
const N_LEDS: usize = 49;
const BLACK_LEVEL: f32 = 0.2;

const BASS_FREQS: Range<f32> = 20.0..330.0;
const MID_FREQS: Range<f32> = 330.0..2600.0;
const HIGH_FREQS: Range<f32> = 2600.0..5500.0;

const BASS_BINS: Range<usize> =
    freq_to_bin(BASS_FREQS.start) as usize..freq_to_bin(BASS_FREQS.end) as usize;
const MID_BINS: Range<usize> =
    freq_to_bin(MID_FREQS.start) as usize..freq_to_bin(MID_FREQS.end) as usize;
const HIGH_BINS: Range<usize> =
    freq_to_bin(HIGH_FREQS.start) as usize..freq_to_bin(HIGH_FREQS.end) as usize;

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
    let mut frames_to_colors = FramesToColors::new();
    println!("");
    loop {
        let samples = samples_rx.recv().expect("error receiving on channel");
        let frames = samples.chunks(N_CHANNELS).map(|w| [w[0], w[1]]);
        for frame in frames {
            buf[buf_i] = frame;
            buf_i += 1;
            if buf_i == FRAMES_PER_PERIOD {
                frames_to_colors.convert_and_send(&buf, &led_tx);
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
        loop {
            let color = rx.recv().expect("Error receiving RGB value on channel");
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

struct FramesToColors {
    max_bass: f32,
    max_mid: f32,
    max_high: f32,
    prev_print_t: time::Instant,
    prev_rgb: Srgb<f32>,
}

impl FramesToColors {
    fn new() -> Self {
        FramesToColors {
            max_bass: 0.01,
            max_mid: 0.01,
            max_high: 0.01,
            prev_print_t: time::Instant::now(),
            prev_rgb: Srgb::new(0.0, 0.0, 0.0),
        }
    }

    fn convert_and_send(
        &mut self,
        stereo_data: &[[f32; 2]; FRAMES_PER_PERIOD],
        led_tx: &mpsc::SyncSender<Rgb8>,
    ) {
        let bin_amps = stereo_frames_to_bin_amps(stereo_data);
        let bass = &bin_amps[BASS_BINS];
        let mids = &bin_amps[MID_BINS];
        let highs = &bin_amps[HIGH_BINS];
        let avg_bass = average(bass.iter().cloned());
        let avg_mid = average(mids.iter().cloned());
        let avg_high = average(highs.iter().cloned());
        let bass_lvl = normalize_amplitude(avg_bass, &mut self.max_bass);
        let mid_lvl = normalize_amplitude(avg_mid, &mut self.max_mid);
        let high_lvl = normalize_amplitude(avg_high, &mut self.max_high);
        let red = bass_lvl.powf(1.8);
        let green = mid_lvl.powf(1.1);
        let blue = high_lvl.powf(1.2);
        let rgb = Srgb::new(red, green, blue);
        let smoothed = self.smooth_color(rgb);
        let saturated = {
            let desat = Hsv::from(smoothed);
            let desaturated_saturation = desat.saturation;
            let saturated_saturation = desaturated_saturation.powf(0.2);
            let sat = Hsv::new(desat.hue, saturated_saturation, desat.value);
            Rgb::from(sat)
        };
        let rgb8 = saturated.into_format::<u8>();
        led_tx
            .send(rgb8)
            .expect("Error sending RGB value on channel");
        self.decay_max_amps();
        self.print_status(bass_lvl, mid_lvl, high_lvl);
    }

    fn smooth_color(&mut self, to: Srgb<f32>) -> Srgb<f32> {
        let from = self.prev_rgb;
        let r = 0.4 * from.red + 0.6 * to.red;
        let g = 0.7 * from.green + 0.3 * to.green;
        let b = 0.5 * from.blue + 0.5 * to.blue;
        let new = Srgb::new(r, g, b);
        self.prev_rgb = new;
        new
    }

    fn decay_max_amps(&mut self) {
        let min_bass = 0.01;
        self.max_bass = (self.max_bass * MAX_AMPS_DECAY_PER_PERIOD).max(min_bass);
        let min_mid = self.max_bass / 4.5;
        self.max_mid = (self.max_mid * MAX_AMPS_DECAY_PER_PERIOD).max(min_mid);
        let min_high = self.max_mid / 2.0;
        self.max_high = (self.max_high * MAX_AMPS_DECAY_PER_PERIOD).max(min_high);
    }

    fn print_status(&mut self, bass_lvl: f32, mid_lvl: f32, high_lvl: f32) {
        let dt = self.prev_print_t.elapsed();
        if dt > time::Duration::from_millis(100) {
            self.prev_print_t = time::Instant::now();
            print!(
                "\rbass: {:4.2}*{:5.2}, mid: {:4.2}*{:5.2}, high: {:4.2}*{:5.2}",
                bass_lvl, self.max_bass, mid_lvl, self.max_mid, high_lvl, self.max_high
            );
            io::stdout().flush().expect("Error flushing stdout");
        }
    }
}

/// bin == index of fft where there is a corresponding frequency
fn stereo_frames_to_bin_amps(
    stereo_frames: &[[f32; 2]; FRAMES_PER_PERIOD],
) -> [f32; (FRAMES_PER_PERIOD >> 1) + 1] {
    let mut mono_samples = [0.0; FRAMES_PER_PERIOD];
    for (i, &[l, r]) in stereo_frames.iter().enumerate() {
        mono_samples[i] = (l + r) / 2.0;
    }
    let mut fft = Fft::new(FRAMES_PER_PERIOD);
    let bin_amps_complex = fft.forward(&mono_samples[..]);
    let mut bin_amps = [0.0f32; (FRAMES_PER_PERIOD >> 1) + 1];
    for (i, &c) in bin_amps_complex.iter().enumerate() {
        bin_amps[i] = (c.re.powi(2) + c.im.powi(2)).sqrt();
    }
    bin_amps
}

const fn bin_to_freq(i: f32) -> f32 {
    (i * SAMPLE_RATE as f32) / FRAMES_PER_PERIOD as f32
}

const fn freq_to_bin(f: f32) -> f32 {
    f * FRAMES_PER_PERIOD as f32 / SAMPLE_RATE as f32
}

fn average<T, I: IntoIterator<Item = T>>(xs: I) -> T
where
    T: Add<T, Output = T>,
    T: Div<T, Output = T>,
    T: One,
{
    let mut it = xs.into_iter();
    let mut sum = it.next()
        .expect("Can't calculate average of empty sequence");
    let mut n = T::one();
    for x in it {
        sum = sum + x;
        n = n + T::one();
    }
    sum / n
}

/// Normalize a frequency amplitude to [0, 1]
fn normalize_amplitude(amp: f32, max_amp: &mut f32) -> f32 {
    let amp_abs = amp.abs();
    if amp_abs > *max_amp {
        *max_amp = amp_abs;
    }
    amp_abs / *max_amp
}

// /// Go faster towards light, slower towards dark
// fn smooth_color(from: Rgb8, to: Rgb8) -> Rgb8 {
//     let from_hsv = Hsv::from(from.into_format::<f32>());
//     let to_hsv = Hsv::from(to.into_format::<f32>());
//     let from_hue = from_hsv.hue.to_positive_degrees();
//     let to_hue = to_hsv.hue.to_positive_degrees();
//     let hue_diff = to_hue - from_hue;
//     let hue = from_hue + 0.04 * hue_diff;
//     let sat = 1.0;
//     let from_val = from_hsv.value;
//     let val_diff = to_hsv.value - from_val;
//     let val = from_val + 0.2 * val_diff;
//     Rgb::from(Hsv::new(hue, sat, val)).into_format::<u8>()
// }

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
