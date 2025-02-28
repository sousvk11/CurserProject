import os

# Create necessary directories
directories = ['Training-doc', 'models', 'utils']
for dir in os.path.join(os.getcwd(), dir) for dir in directories:
    os.makedirs(dir, exist_ok=True)

# Create web-url.txt file
with open('web-url.txt', 'w') as f:
    f.write('# Add your training URLs here, one per line\n') 