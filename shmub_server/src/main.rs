extern crate cpal;
extern crate shmub_common;
extern crate std_semaphore;

use cpal::{Format, OutputBuffer, SampleFormat, SampleRate, StreamData, UnknownTypeOutputBuffer};
use shmub_common::*;
use std::collections::VecDeque;
use std::net::{Ipv4Addr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::thread;
use std_semaphore::Semaphore;

const PORT: u16 = 14320;
const N_BUFFERED_SAMPLES: usize = 1024;

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
    let buffer = AudioBuffer::new(N_BUFFERED_SAMPLES);
    let buffer2 = buffer.clone();

    thread::spawn(move || {
        event_loop.run(move |_, data| {
            let mut out_buf = get_i16_buffer(as_output_buffer(data));
            for out_sample in out_buf.chunks_mut(N_CHANNELS) {
                let sample = buffer2.pop_front();
                for (out, &value) in out_sample.iter_mut().zip(&sample[..]) {
                    *out = value;
                }
            }
        });
    });

    loop {
        let packet = recv_packet(&socket);
        for sample in packet.samples()[..].iter().cloned() {
            buffer.try_push_back(sample);
        }
    }
}

/// A thread-safe buffer of samples that handles buffering in a way that makes sense for audio
#[derive(Clone)]
struct AudioBuffer {
    guarded_queue: Arc<Mutex<VecDeque<Sample>>>,
    optimal_size: usize,
    available_samples: Arc<Semaphore>,
}

impl AudioBuffer {
    fn new(optimal_size: usize) -> Self {
        assert!(optimal_size > 1, "Optimal size must be greater than 1");
        AudioBuffer {
            guarded_queue: Arc::new(Mutex::new(VecDeque::with_capacity(optimal_size * 2))),
            optimal_size,
            available_samples: Arc::new(Semaphore::new(0)),
        }
    }

    /// If the buffer is not unbuffering to `optimal_size`, push the sample
    ///
    /// Returns whether the sample was pushed
    fn try_push_back(&self, sample: Sample) -> bool {
        let mut queue = self
            .guarded_queue
            .lock()
            .expect("Error locking queue mutex");
        if queue.len() < self.optimal_size * 2 {
            queue.push_back(sample);
            self.available_samples.release();
            true
        } else {
            false
        }
    }

    fn wait_for_buffering(&self) {
        for _ in 0..self.optimal_size {
            self.available_samples.acquire();
        }
        for _ in 0..self.optimal_size {
            self.available_samples.release();
        }
    }

    /// If the buffer is not buffering to `optimal_size`, pop a sample
    fn pop_front(&self) -> Sample {
        self.available_samples.acquire();
        let (sample, wait) = {
            let mut queue = self
                .guarded_queue
                .lock()
                .expect("error locking queue mutex");
            let sample = queue.pop_front().expect("no element to pop queue");
            let wait = queue.len() == 1;
            (sample, wait)
        };
        if wait {
            self.wait_for_buffering()
        }
        sample
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
