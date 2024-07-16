use std::fs::File;
use std::io::{self, Write, Read, BufReader, BufWriter, BufRead};
use log::debug;
use rayon::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Debug)]
pub enum TupleBuffer {
    InMemory(MemoryBuffer),
    OnDisk(DiskBuffer),
}

impl TupleBuffer {
    pub fn push(&mut self, tuple: u32) {
        match self {
            TupleBuffer::InMemory(buffer) => buffer.push(tuple),
            TupleBuffer::OnDisk(buffer) => buffer.push(tuple),
        }
    }

    pub fn into_iterator(self) -> TupleIterator {
        match self {
            TupleBuffer::InMemory(buffer) => TupleIterator::InMemory(buffer.into_iterator()),
            TupleBuffer::OnDisk(buffer) => TupleIterator::OnDisk(buffer.into_iterator()),
        }
    }
}

#[derive(Debug)]
pub enum TupleIterator {
    InMemory(MemoryIterator),
    OnDisk(DiskIterator),
}

impl Iterator for TupleIterator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TupleIterator::InMemory(iterator) => iterator.next(),
            TupleIterator::OnDisk(iterator) => iterator.next(),
        }
    }
}

#[derive(Debug)]
pub struct Quantiles {
    num_quantiles: usize,
    quantiles: Vec<Vec<u32>>,
}

impl Quantiles {
    pub fn new(num_quantiles: usize) -> Self {
        Quantiles {
            num_quantiles,
            quantiles: Vec::new(),
        }
    }

    pub fn compute_quantiles(&mut self, run: &[u32]) {
        let run_len = run.len();
        let mut quantile_values = Vec::new();

        for i in 1..self.num_quantiles {
            let quantile_index = (i * run_len) / self.num_quantiles;
            quantile_values.push(run[quantile_index]);
        }

        self.quantiles.push(quantile_values);
    }
}

pub struct ExternalSorting {
    input: Vec<u32>,
    intermediate_buffers: Vec<TupleBuffer>,
    output: TupleBuffer,
    run_size: usize,
    quantiles: Quantiles,
}

// Manual implementation of Debug for ExternalSorting
impl std::fmt::Debug for ExternalSorting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalSorting")
            .field("intermediate_buffers", &self.intermediate_buffers)
            .field("output", &self.output)
            .field("run_size", &self.run_size)
            .field("quantiles", &self.quantiles)
            .finish()
    }
}

impl ExternalSorting {
    pub fn execute(&mut self) {
        // Step 1: Read input and create sorted runs in parallel
        let num_runs = (self.input.len() + self.run_size - 1) / self.run_size;
        let mut runs: Vec<Vec<u32>> = (0..num_runs)
            .into_par_iter()
            .map(|i| {
                let start = i * self.run_size;
                let end = (start + self.run_size).min(self.input.len());
                let mut run: Vec<u32> = self.input[start..end].to_vec();
                run.sort();
                debug!("run added {:?}", run);
                run
            })
            .collect();

        // Compute quantiles for each run
        for run in &runs {
            self.quantiles.compute_quantiles(run);
        }

        for run in runs {
            self.intermediate_buffers.push(TupleBuffer::InMemory(MemoryBuffer::from_vec(run)));
        }

        debug!("Intermediate buffers len: {}", self.intermediate_buffers.len());

        // Step 2: Merge sorted runs
        while self.intermediate_buffers.len() > 1 {
            let mut new_buffers = Vec::new();
            let mut iter = std::mem::take(&mut self.intermediate_buffers).into_iter();

            while let Some(first) = iter.next() {
                if let Some(second) = iter.next() {
                    let merged_buffer = self.merge_buffers(first, second);
                    new_buffers.push(merged_buffer);
                } else {
                    new_buffers.push(first);
                }
            }

            self.intermediate_buffers = new_buffers;
        }

        // Step 3: Output the final sorted buffer
        if let Some(final_buffer) = self.intermediate_buffers.pop() {
            for tuple in final_buffer.into_iterator() {
                self.output.push(tuple);
            }
        }
    }

    fn merge_buffers(&self, buffer1: TupleBuffer, buffer2: TupleBuffer) -> TupleBuffer {
        let mut merged_buffer = MemoryBuffer::new();
        let mut iter1 = buffer1.into_iterator();
        let mut iter2 = buffer2.into_iterator();

        let mut next1 = iter1.next();
        let mut next2 = iter2.next();

        while next1.is_some() || next2.is_some() {
            match (next1, next2) {
                (Some(tuple1), Some(tuple2)) => {
                    if tuple1 < tuple2 {
                        merged_buffer.push(tuple1);
                        next1 = iter1.next();
                    } else {
                        merged_buffer.push(tuple2);
                        next2 = iter2.next();
                    }
                }
                (Some(tuple1), None) => {
                    merged_buffer.push(tuple1);
                    next1 = iter1.next();
                }
                (None, Some(tuple2)) => {
                    merged_buffer.push(tuple2);
                    next2 = iter2.next();
                }
                (None, None) => break,
            }
        }

        TupleBuffer::InMemory(merged_buffer)
    }
}

#[derive(Debug)]
pub struct MemoryBuffer {
    data: Vec<u32>,
}

impl MemoryBuffer {
    pub fn new() -> Self {
        MemoryBuffer { data: Vec::new() }
    }

    pub fn push(&mut self, tuple: u32) {
        self.data.push(tuple);
    }

    pub fn into_iterator(self) -> MemoryIterator {
        MemoryIterator {
            data: self.data.into_iter(),
        }
    }

    pub fn from_vec(data: Vec<u32>) -> Self {
        MemoryBuffer { data }
    }
}

#[derive(Debug)]
pub struct MemoryIterator {
    data: std::vec::IntoIter<u32>,
}

impl Iterator for MemoryIterator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.data.next()
    }
}

#[derive(Debug)]
pub struct DiskBuffer {
    file_path: String,
}

impl DiskBuffer {
    pub fn new(file_path: String) -> Self {
        DiskBuffer { file_path }
    }

    pub fn push(&mut self, tuple: u32) {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.file_path)
            .expect("Unable to open file");
        writeln!(file, "{}", tuple).expect("Unable to write data to file");
    }

    pub fn into_iterator(self) -> DiskIterator {
        DiskIterator {
            file: std::fs::File::open(self.file_path).expect("Unable to open file"),
        }
    }
}

#[derive(Debug)]
pub struct DiskIterator {
    file: std::fs::File,
}

impl Iterator for DiskIterator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buffer = String::new();
        if let Ok(bytes_read) = self.file.read_to_string(&mut buffer) {
            if bytes_read == 0 {
                None
            } else {
                Some(buffer.trim().parse().expect("Unable to parse number"))
            }
        } else {
            None
        }
    }
}

fn read_input_from_file(file_path: &str) -> io::Result<Vec<u32>> {
    let file = File::open(file_path)?;
    let buf_reader = BufReader::new(file);
    let mut input = Vec::new();
    for line in buf_reader.lines() {
        let number: u32 = line?.parse().expect("Unable to parse number");
        input.push(number);
    }
    Ok(input)
}

fn write_input_to_file(file_path: &str, data: &[u32]) -> io::Result<()> {
    let file = File::create(file_path)?;
    let mut buf_writer = BufWriter::new(file);
    for &value in data {
        writeln!(buf_writer, "{}", value)?;
    }
    Ok(())
}

fn generate_random_input(size: usize) -> Vec<u32> {
    let mut rng = thread_rng();
    let mut data: Vec<u32> = (1..=size as u32).collect();
    data.shuffle(&mut rng);
    data
}

fn verify_sorted(output: &[u32]) -> bool {
    for i in 1..output.len() {
        if output[i - 1] > output[i] {
            return false;
        }
    }
    true
}

fn main() {
    // Initialize logger
    env_logger::init();

    let file_path = "1,000.txt";
    // let input_data = generate_random_input(10_000);
    // write_input_to_file(&file_path, &input_data).expect("Unable to write input file");
    // Read input from file
    let input_data = read_input_from_file(&file_path).expect("Unable to read input file");

    // Create output buffer
    let output_buffer = MemoryBuffer::new();

    // Create ExternalSorting object
    let mut sorter = ExternalSorting {
        input: input_data,
        intermediate_buffers: Vec::new(),
        output: TupleBuffer::InMemory(output_buffer),
        run_size: 100, // Set run size as needed
        quantiles: Quantiles::new(10), // Set number of quantiles as needed
    };
    // Execute sorting
    sorter.execute();

    // Retrieve sorted output
    if let TupleBuffer::InMemory(output) = sorter.output {
        let sorted_output: Vec<u32> = output.data;
        if verify_sorted(&sorted_output) {
            debug!("sorted output: {:?}", sorted_output);
            println!("The output is correctly sorted.");
        } else {
            println!("The output is NOT correctly sorted.");
            debug!("Sorted output: {:?}", sorted_output);
            println!("Sorted output: {:?}", sorted_output);
        }
    }

    // Print quantiles
    println!("Quantiles: {:?}", sorter.quantiles.quantiles);
}