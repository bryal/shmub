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
const BUFFER_LATENCY_MS: usize = 45;
const BUFFER_LATENCY_FRAMES: usize = (BUFFER_LATENCY_MS * SAMPLE_RATE as usize) / 1000;
const SEQ_RESTART_MARGIN: u32 = 100;

fn main() {
    println!("Shmub Audio Server");
    println!("Receive streamed UDP audio data and output to connected sound device.");
    println!(
        "Buffer size: {} frames; Restart margin: {} frames",
        BUFFER_LATENCY_FRAMES, SEQ_RESTART_MARGIN
    );
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
    println!("Listening to port {}", PORT);

    // Sample channel
    let buffer = AudioBuffer::new(BUFFER_LATENCY_FRAMES);
    let buffer2 = buffer.clone();

    thread::spawn(move || {
        let mut prev_frame = [0, 0];
        event_loop.run(move |_, data| {
            let mut out_buf = get_i16_buffer(as_output_buffer(data));
            for out_frame in out_buf.chunks_mut(N_CHANNELS) {
                let frame = buffer2.try_pop_front().unwrap_or(prev_frame);
                for (out, &value) in out_frame.iter_mut().zip(&frame[..]) {
                    *out = value;
                }
                prev_frame = frame;
            }
            buffer2.wait_for_buffering_if_empty();
        });
    });

    let mut last_seq_index = 0;
    loop {
        let packet = recv_packet(&socket);
        let is_newer = packet.seq_index > last_seq_index;
        let probably_reset = packet.seq_index < last_seq_index.saturating_sub(SEQ_RESTART_MARGIN);
        if is_newer || probably_reset {
            if probably_reset {
                println!(
                    "probably reset. packet: {}, last: {}",
                    packet.seq_index, last_seq_index
                );
            }
            for sample in packet.samples[..].iter().cloned() {
                buffer.try_push_back(sample);
            }
            last_seq_index = packet.seq_index;
        } else {
            println!(
                "out of order packet. this: {}, last: {}",
                packet.seq_index, last_seq_index
            );
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
        let mut queue = self.guarded_queue
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
        println!("buffering...");
        for _ in 0..self.optimal_size {
            self.available_samples.acquire();
        }
        for _ in 0..self.optimal_size {
            self.available_samples.release();
        }
        println!("buffered!");
    }

    fn wait_for_buffering_if_empty(&self) {
        let wait = {
            let queue = self.guarded_queue
                .lock()
                .expect("error locking queue mutex");
            let wait = queue.len() <= 10;
            wait
        };
        if wait {
            self.wait_for_buffering()
        }
    }

    /// If the buffer is not buffering to `optimal_size`, pop a sample
    fn pop_front(&self) -> Sample {
        self.available_samples.acquire();
        let (sample, wait) = {
            let mut queue = self.guarded_queue
                .lock()
                .expect("error locking queue mutex");
            let sample = queue.pop_front().expect("no element to pop queue");
            let wait = queue.len() <= 1;
            (sample, wait)
        };
        if wait {
            self.wait_for_buffering()
        }
        sample
    }

    fn try_pop_front(&self) -> Option<Sample> {
        let mut queue = self.guarded_queue
            .lock()
            .expect("error locking queue mutex");
        if queue.len() > 0 {
            queue.pop_front()
        } else {
            None
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
