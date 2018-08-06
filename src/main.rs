extern crate cpal;
extern crate shmub_common;

use cpal::{Format, OutputBuffer, SampleFormat, SampleRate, StreamData, UnknownTypeOutputBuffer};
use shmub_common::*;
use std::net::{Ipv4Addr, UdpSocket};
use std::sync::mpsc;
use std::thread;

const PORT: u16 = 14320;
const CHANNEL_N_BUFFERED_SAMPLES: usize = 1024;

fn main() {
    println!("Shmub Audio Server");
    println!("Receive streamed UDP audio data and output to connected sound device.");
    println!("");
    let device = prompt_device();
    let format = Format {
        channels: N_CHANNELS as u16,
        sample_rate: SampleRate(SAMPLE_RATE),
        data_type: SampleFormat::I16,
    };
    let event_loop = cpal::EventLoop::new();
    let stream_id = event_loop
        .build_output_stream(&device, &format)
        .expect(&format!(
            "Failed to build output stream for device {} with format {:?}",
            device.name(),
            format
        ));
    event_loop.play_stream(stream_id);
    let socket = UdpSocket::bind((Ipv4Addr::new(0, 0, 0, 0), PORT)).expect("Failed to open socket");
    // Sample channel
    let (tx, rx) = mpsc::sync_channel::<[i16; N_CHANNELS]>(CHANNEL_N_BUFFERED_SAMPLES);
    thread::spawn(move || {
        event_loop.run(move |_, data| {
            let mut out_buf = get_i16_buffer(as_output_buffer(data));
            for out_sample in out_buf.chunks_mut(N_CHANNELS) {
                let sample = rx.recv().expect("Error receiving sample");
                for (out, &value) in out_sample.iter_mut().zip(&sample[..]) {
                    *out = value;
                }
            }
        });
    });
    loop {
        let packet = recv_packet(&socket);
        for sample in packet.samples()[..].iter().cloned() {
            match tx.try_send(sample) {
                Ok(_) | Err(mpsc::TrySendError::Full(_)) => (),
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    panic!("Lost connection to output thread")
                }
            }
        }
    }
}

fn get_i16_buffer<'b>(unknown_buf: UnknownTypeOutputBuffer<'b>) -> OutputBuffer<'b, i16> {
    match unknown_buf {
        UnknownTypeOutputBuffer::I16(buf) => buf,
        _ => panic!("Expected I16 buffer"),
    }
}

fn as_output_buffer(data: StreamData) -> UnknownTypeOutputBuffer {
    match data {
        StreamData::Output { buffer } => buffer,
        _ => panic!("Expected output data buffer"),
    }
}

fn recv_packet(socket: &UdpSocket) -> Packet {
    let mut buf = [0; PACKET_SIZE];
    socket
        .recv_from(&mut buf[..])
        .expect("Error receiving from socket");
    Packet::parse(&buf[..]).expect("Error parsing packet")
}

fn prompt_device() -> cpal::Device {
    println!("Select audio output device:");
    let mut devices = cpal::output_devices().collect::<Vec<_>>();
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
