use llama_cpp_rs::{options::{ModelOptions, PredictOptions}, LLama,};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() {

    let model_options = ModelOptions::default();
    let token_counter = Arc::new(Mutex::new(0));
    let token_counter_clone = Arc::clone(&token_counter);

    let llama = LLama::new("./models/wizvic7bQ3KM.gguf".into(), &model_options).unwrap();
    let predict_options = PredictOptions {
        tokens: 0,
        threads: 16,
        top_k: 90,
        top_p: 0.86,
        token_callback: Some(Box::new(move |token| {

            print!("{}", token);
            io::stdout().flush().unwrap();

            let mut counter = token_counter.lock().unwrap();
            *counter += 1;

            true
        })),
        ..Default::default()
    };

    let mut input = String::new();
    print!(">.> ");
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut input).unwrap();
    let prompt = format!("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input} ASSISTANT:");

    let start = Instant::now();
    llama.predict(prompt.into(), predict_options).unwrap();
    let elapsed = start.elapsed();
    println!("\n. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .");
    println!("INFO: Total tokens generated: {}", token_counter_clone.lock().unwrap());
    println!("INFO: Total time elapsed: {}", elapsed.as_secs_f64());
    println!("INFO: Total time per token: {:?}", elapsed.as_secs_f64() / *token_counter_clone.lock().unwrap() as f64);
    println!(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .");
}
