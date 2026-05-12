---
title: "Rust 1 : Why Rust for Data Analysis ?"
header :
  image : /assets/images/Rust-4-Data-Analysis.png
  teaser: https://logowik.com/content/uploads/images/rust4784.logowik.com.webp
comments : true
share : true
categories:
  - Rust
---

Hey guys. It's maybe the opportunity to take a challenge on myself to try this one. I found it difficult really to understand Rust as a programming language. The syntax itself seems difference compared to Python, SQL, or C++. Compared to Go, it seems similar they say. I wanted to create a series while I am also learning Rust, I'd like to compare it on how we usually do it in Python. So I can learn a new thing while keep reviewing again my Python skills.

If you have spent years wrangling data in Python—cleaning CSVs with pandas, plotting with matplotlib, and training models with scikit-learn—you already know the joy of expressive, high-level data tools. It's also insanely easy to understand and very intuitive. But when you spent a lot of time using Python and try apply your knowledge in the real huge dataset, you have also likely felt the pain: a script that takes hours to process a 10 GB dataset, a memory leak in a production pipeline, or a type error that only surfaces after thirty minutes of runtime.

Rust is a systems programming language designed to give you C++-like performance with memory safety guarantees enforced at compile time. For data analysis, this means:

- **Performance**: Rust compiles to native machine code. Polars, a DataFrame library written in Rust, routinely outperforms pandas by 5×–50× on single machines.
- **Memory safety**: No garbage collector, no segfaults, and no data races. The borrow checker catches memory bugs before your code ever runs.
- **Production reliability**: Rust's strict type system and Result-based error handling make data pipelines that are robust by default.
- **Growing ecosystem**: Crates like `polars`, `ndarray`, `linfa`, and `plotters` are maturing rapidly.

So, in short, The primary purpose of using Rust is enhanced safety, speed, and concurrency, or the ability to run multiple computations simultaneously.

# Rust Advantages
## C-like Speed
Rust has been developed to offer lightning-fast performance similar to the C programming language. In addition, it provides the added advantages of memory and thread safety. o illustrate this point further, consider the following code snippet, which efficiently calculates the Fibonacci sequence using Rust:

```rust
use std::hint::black_box;

fn fibonacci(n: u64) -> u64 {
    match n {
        1 | 0 => n,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn main() {
    let mut total: f64 = 0.0;
    for _ in 1..=20 {
        let start = std::time::Instant::now();
        black_box(fibonacci(black_box(40)));
        let elapsed = start.elapsed().as_secs_f64();
        total += elapsed;
    }
    let avg_time = total / 20.0;
    println!("Average time taken: {} s", avg_time);
}

// Average time taken: 0.3688494305 s
```

The above code snippet calculates the 40th number in the Fibonacci sequence using recursion. **It executes in less than a second**, much faster than equivalent code in many other languages. Consider Python, for example. **It took approximately 22.2 seconds in Python to calculate the same Fibonacci sequence**, which is way slower than the Rust version.

```python
>>> import timeit
>>> def fibonacci(n):
... if n < 2:
... return n
... return fibonacci(n-1) + fibonacci(n-2)
... 
>>> timeit.Timer("fibonacci(40)", "from __main__ import fibonacci").timeit(number=1)
22.262923367998155
```

## 2. Type Safety.

Rust is designed to catch many errors at compile time rather than runtime, reducing the likelihood of bugs in the final product. Take the following example of Rust code that demonstrates its type safety:

```rust
fn add_numbers(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let a = 1;
    let b = "2";
    let sum = add_numbers(a, b); // Compile error:  expected `i32`, found `&amp;str
    println!("{} + {} = {}", a, b, sum);
}
```

The above code snippet attempts to add an integer and a string together, which is not allowed in Rust due to type safety. The code fails to compile with a helpful error message that points to the problem.

## 3. Memory safety.

Rust has been meticulously developed to prevent prevalent memory errors, including buffer overflows and null pointer dereferences, thereby reducing the probability of security vulnerabilities. This is exemplified by the following scenario that showcases Rust’s memory safety measures:

```rust
fn main() {
    let mut v = vec![1, 2, 3];
    let first = v.get(0); // Compile error: immutable borrow occurs here
    v.push(4); // Compile error: mutable borrow occurs here
    println!("{:?}", first); // Compile error: immutable borrow later used here
}
```

The above code attempts to append an element to a vector while holding an immutable reference to its first element. This is not allowed in Rust due to memory safety, and the code fails to compile with a helpful error message.

## 4. True and safe parallelism.

The ownership model of Rust provides a secure and proficient means of parallelism, eliminating data races and other bugs related to concurrency. An illustrative example of Rust’s parallelism is presented below:

```rust
use std::thread;

fn main() {
    let mut handles = vec![];
    let mut x = 0;
    for i in 0..10 {
        handles.push(thread::spawn(move || {
            x += 1;
            println!("Hello from thread {} with x = {}", i, x);
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

// Output

// Hello from thread 0 with x = 1
// Hello from thread 1 with x = 1
// Hello from thread 2 with x = 1
// Hello from thread 4 with x = 1
// Hello from thread 3 with x = 1
// Hello from thread 5 with x = 1
// Hello from thread 6 with x = 1
// Hello from thread 7 with x = 1
// Hello from thread 8 with x = 1
// Hello from thread 9 with x = 1

The above code creates ten threads that print messages to the console. Rust’s ownership model guarantees that each thread has exclusive access to the necessary resources, effectively preventing data races and other concurrency-related bugs.
```

## 5. Rich Ecosystem.

Rust offers a thriving and dynamic ecosystem with diverse libraries and tools catering to a wide range of domains. For instance, Rust provides powerful data analysis tools such as [ndarray](https://docs.rs/ndarray/latest/ndarray/) and [polors](https://www.pola.rs/), and its [serde](https://serde.rs/) library outperforms any JSON library written in Python.

These advantages and others make Rust an attractive option for developers such as data scientists seeking a convenient programming language that equips them with an extensive list of tools.

# TL;DR
Rust stands out as a practical choice in data science due to its exceptional performance and persistent security features. While it may not possess all the bells and whistles that Python does, Rust offers outstanding efficiency when handling large datasets. Additionally, developers can use an array of libraries explicitly designed for data analysis to streamline their workflow further. With proper mastery of this language’s complexities, those working within the field can gain significant advantages by incorporating Rust into their toolkit.
