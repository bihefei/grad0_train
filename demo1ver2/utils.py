def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 4:
                texts.append(parts[3])
                labels.append(parts[2])
    return texts, labels