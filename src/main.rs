extern crate cpal;
extern crate shmub_common;

use cpal::{Format, Sample, SampleFormat, SampleRate, StreamData, UnknownTypeInputBuffer};
use shmub_common::*;
use std::io::{self, BufRead, Write};
use std::net::{Ipv4Addr, UdpSocket};

const SERVER_PORT: u16 = 14320;
const CLIENT_PORT: u16 = 14321;

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

    let socket =
        UdpSocket::bind((Ipv4Addr::new(0, 0, 0, 0), CLIENT_PORT)).expect("Failed to open socket");
    let server = (Ipv4Addr::new(127, 0, 0, 1), SERVER_PORT);

    let mut buf = [[0i16; N_CHANNELS]; PACKET_N_PCM_SAMPLES];
    let mut buf_i = 0usize;
    event_loop.run(move |_, data| {
        let samples = to_i16_buffer(as_input_buffer(&data))
            .chunks(N_CHANNELS)
            .map(|w| [w[0], w[1]])
            .collect::<Vec<_>>();
        let mut i = 0;
        loop {
            if buf_i == PACKET_N_PCM_SAMPLES {
                let packet = Packet::new(&buf).expect("Error creating packet");
                socket
                    .send_to(&packet.to_bytes()[..], server)
                    .expect("Error sending packet");
                buf_i = 0;
            } else if i == samples.len() {
                break;
            } else {
                buf[buf_i] = samples[i];
                buf_i += 1;
                i += 1;
            }
        }
    });
}

fn to_i16_buffer(unknown_buf: &UnknownTypeInputBuffer) -> Vec<i16> {
    match unknown_buf {
        UnknownTypeInputBuffer::U16(buf) => buf.iter().map(Sample::to_i16).collect(),
        UnknownTypeInputBuffer::F32(buf) => buf.iter().map(Sample::to_i16).collect(),
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
