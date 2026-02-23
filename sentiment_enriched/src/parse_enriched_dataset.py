def parse_dataset(filepath):
    texts, slots, classes = [], [], []
    with open(filepath, 'r', encoding='utf-8') as file:
        current_text, current_slots, current_classes = [], [], []
        text_id = 0 
        for line in file:
            line = line.strip()
            if line.startswith('# sent_id'):
              sent_text = line.split()[3]
              sent, text = sent_text.split('_')
              if text_id != int(text):
                texts.append(current_text)
                slots.append(current_slots)
                classes.append(current_classes)
                text_id = int(text)
                current_text, current_slots, current_classes = [], [], []
            elif line.startswith('# text'):
              continue
            elif not line:
              continue
            else:
                token = line.split('\t')
                current_text.append(token[1])
                current_classes.append(token[-1])
                current_slots.append(token[-2])
        if current_text:
          texts.append(current_text)
          slots.append(current_slots)
          classes.append(current_classes)
    return texts, slots, classes

def extract_classes_and_slots(filepath1):
    classes = set()
    slots = set()
    with open(filepath1, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) < 11:
                continue
            class_field = fields[11]
            slot_field = fields[10]
            classes.add(class_field)
            slots.add(slot_field)
    return classes, slots

def classes_slots_dicts(filepath1, filepath2):
    train_sets = extract_classes_and_slots(filepath1)
    val_sets = extract_classes_and_slots(filepath2)
    classes = set(train_sets[0]).union(set(val_sets[0]))
    slots = set(train_sets[1]).union(set(val_sets[1]))

    classes = sorted(classes)
    slots = sorted(slots)
    classes.append('PAD')
    slots.append('PAD')

    class2idx = {cls: idx for idx, cls in enumerate(classes)}
    idx2class = {idx: cls for cls, idx in class2idx.items()}

    slot2idx = {slot: idx for idx, slot in enumerate(slots)}
    idx2slot = {idx: slot for slot, idx in slot2idx.items()}

    return {
        'classes': classes,
        'slots': slots,
        'class2idx': class2idx,
        'idx2class': idx2class,
        'slot2idx': slot2idx,
        'idx2slot': idx2slot
    }