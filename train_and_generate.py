from utils.document_processor import DocumentProcessor
from utils.web_scraper import WebScraper
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CodeGenerationModel:
    def __init__(self):
        try:
            logging.info("Initializing model and tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"Using device: {self.device}")
            self.model.to(self.device)
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            sys.exit(1)
    
    def prepare_training_data(self):
        try:
            logging.info("Processing documents...")
            doc_processor = DocumentProcessor('Training-doc')
            documents = doc_processor.process_all_documents()
            
            logging.info("Processing web content...")
            web_scraper = WebScraper('web-url.txt')
            web_content = web_scraper.scrape_all_urls()
            
            training_data = documents + web_content
            if not training_data:
                logging.warning("No training data collected!")
                return False
                
            logging.info(f"Collected {len(training_data)} documents/pages")
            with open('combined_training_data.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(training_data))
            return True
        except Exception as e:
            logging.error(f"Error preparing training data: {str(e)}")
            return False
    
    def train(self, epochs=3, batch_size=4):
        try:
            logging.info("Setting up training arguments...")
            training_args = TrainingArguments(
                output_dir="./models",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=500,
                save_total_limit=2,
                logging_dir="./logs",
                logging_steps=10,
            )
            
            logging.info("Creating dataset...")
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path="combined_training_data.txt",
                block_size=128
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            logging.info("Starting training...")
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )
            
            trainer.train()
            
            logging.info("Saving model...")
            self.model.save_pretrained('./models/final_model')
            self.tokenizer.save_pretrained('./models/final_model')
            return True
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            return False
    
    def generate_code(self, prompt: str, max_length: int = 200) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model = CodeGenerationModel()
    
    logging.info("Starting data preparation...")
    if not model.prepare_training_data():
        logging.error("Failed to prepare training data")
        return
    
    logging.info("Starting model training...")
    if not model.train():
        logging.error("Failed to train model")
        return
    
    logging.info("Starting generation phase...")
    while True:
        try:
            prompt = input("Enter your prompt (or 'quit' to exit): ")
            if prompt.lower() == 'quit':
                break
            generated_code = model.generate_code(prompt)
            print("\nGenerated Code:")
            print(generated_code)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.error(f"Error during code generation: {str(e)}")

if __name__ == "__main__":
    main() 