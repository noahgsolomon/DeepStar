import { HfInference } from "@huggingface/inference";

const hf = new HfInference(process.env.API_KEY);

const model = await hf.textGeneration({
  inputs: "What is bayesian probability?",
  model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
});

const text = model.generated_text;

console.log(text);
