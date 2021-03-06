#![feature(duration_float, type_ascription)]

extern crate cpal;
extern crate shmub_common;

use cpal::{Format, SampleFormat, SampleRate, StreamData, UnknownTypeInputBuffer};
use shmub_common::*;
use std::net::{Ipv4Addr, UdpSocket};
use std::sync::mpsc;
use std::{thread, time};

const SERVER_PORT: u16 = 14320;
const CLIENT_PORT: u16 = 14321;

struct Config {
    server_ip: Ipv4Addr,
    server_port: u16,
    client_port: u16,
}

fn main() {
    println!("Shmub Audio Client");
    println!("Capture audio from an input device and stream to a Shmub server");

    let config = match parse_args() {
        Ok(c) => c,
        Err(exe) => {
            println!("Usage: {} SERVER_IP SERVER_PORT CLIENT_PORT", exe);
            return;
        }
    };

    let device = prompt_device();
    let format = Format {
        channels: N_CHANNELS as u16,
        sample_rate: SampleRate(SAMPLE_RATE),
        data_type: SampleFormat::I16,
    };
    let event_loop = cpal::EventLoop::new();
    let stream_id = event_loop
        .build_input_stream(&device, &format)
        .expect(&format!(
            "Failed to build input stream for device {} with format {:?}",
            device.name(),
            format
        ));
    event_loop.play_stream(stream_id);

    let socket = UdpSocket::bind((Ipv4Addr::new(0, 0, 0, 0), config.client_port))
        .expect("Failed to open socket");
    let server = (config.server_ip, config.server_port);

    let (tx, rx) = mpsc::sync_channel(4);
    thread::spawn(move || {
        let mut frames_per_s = 0;
        let mut t0 = time::Instant::now();
        event_loop.run(move |_, data| {
            let samples = to_i16_buffer(as_input_buffer(&data));
            let len = samples.len();
            frames_per_s += samples.len() / N_CHANNELS;
            match tx.try_send(samples) {
                Err(mpsc::TrySendError::Disconnected(_)) => panic!("channel disconnected"),
                _ => (),
            }
            let t1 = time::Instant::now();
            let t = t1.duration_since(t0).as_float_secs();
            if t >= 5.0 {
                t0 = t1;
                println!("Frames per second: {}", frames_per_s as f64 / t);
                frames_per_s = 0;
            }
        })
    });

    let mut buf = [[0: Sample; N_CHANNELS]; PACKET_N_PCM_FRAMES];
    let mut buf_i = 0usize;
    let mut seq_index = 0;
    loop {
        let samples = rx.recv().expect("error receiving on channel");
        let frames = samples.chunks(N_CHANNELS).map(|w| [w[0], w[1]]);
        for frame in frames {
            buf[buf_i] = frame;
            buf_i += 1;
            if buf_i == PACKET_N_PCM_FRAMES {
                let packet = Packet::new(seq_index, &buf).expect("Error creating packet");
                socket
                    .send_to(&packet.to_bytes()[..], server)
                    .expect("Error sending packet");
                seq_index += 1;
                buf_i = 0;
            }
        }
    }
}

fn to_i16_buffer(unknown_buf: &UnknownTypeInputBuffer) -> Vec<i16> {
    match unknown_buf {
        UnknownTypeInputBuffer::U16(buf) => buf.iter().map(cpal::Sample::to_i16).collect(),
        UnknownTypeInputBuffer::F32(buf) => buf.iter().map(cpal::Sample::to_i16).collect(),
        UnknownTypeInputBuffer::I16(buf) => buf.to_vec(),
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

fn parse_args() -> Result<Config, String> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() > 1 {
        let server_ip = args[1].parse().expect("Invalid server IP");
        let mut server_port = SERVER_PORT;
        let mut client_port = CLIENT_PORT;
        if args.len() > 2 {
            server_port = args[2].parse().expect("Invalid server port");
            if args.len() == 4 {
                client_port = args[3].parse().expect("Invalid client port")
            } else {
                println!("Unexpected argument `{}`", args[3]);
                return Err(args[0].clone());
            }
        }
        Ok(Config {
            server_ip,
            server_port,
            client_port,
        })
    } else {
        Err(args[0].clone())
    }
}
