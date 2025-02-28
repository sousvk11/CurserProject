from utils.document_processor import DocumentProcessor
from utils.web_scraper import WebScraper
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

class CodeGenerationModel:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def prepare_training_data(self):
        # Process documents
        doc_processor = DocumentProcessor('Training-doc')
        documents = doc_processor.process_all_documents()
        
        # Process web content
        web_scraper = WebScraper('web-url.txt')
        web_content = web_scraper.scrape_all_urls()
        
        # Combine all training data
        training_data = documents + web_content
        
        # Save combined data to a file
        with open('combined_training_data.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(training_data))
    
    def train(self, epochs=3, batch_size=4):
        training_args = TrainingArguments(
            output_dir="./models",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
        )
        
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path="combined_training_data.txt",
            block_size=128
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        trainer.train()
        self.model.save_pretrained('./models/final_model')
        self.tokenizer.save_pretrained('./models/final_model')
    
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
    
    # Training phase
    print("Preparing training data...")
    model.prepare_training_data()
    print("Training model...")
    model.train()
    
    # Generation phase
    while True:
        prompt = input("Enter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        generated_code = model.generate_code(prompt)
        print("\nGenerated Code:")
        print(generated_code)

if __name__ == "__main__":
    main() 