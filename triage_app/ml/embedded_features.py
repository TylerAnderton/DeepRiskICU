import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch

from .utils import assert_unique

clinicalBERT = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinicalBERT_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", use_fast=True)

# def get_bert_embeddings(texts:list[str], model, tokenizer, batch_size=32, tokenizer_max_length=512, default_embedding_length=768) -> np.ndarray:
#     if torch.cuda.is_available():
#         print('Using CUDA')
#         model = model.cuda()

#     print(f'Getting embeddings for {len(texts)} items...')
    
#     embeddings = []
#     for i in range(0, len(texts), batch_size):
#         batch_idx = i // batch_size
#         if batch_idx % 20 == 0: # Print every 50 batches
#             print(f'Processing batch {batch_idx} of {len(texts) // batch_size}')
            
#         batch_texts = texts[i:i+batch_size]
#         inputs = tokenizer(batch_texts, padding="max_length", truncation=True, 
#                           max_length=tokenizer_max_length, return_tensors="pt")  # Reduced max_length
        
#         # Move to GPU if available
#         if torch.cuda.is_available():
#             inputs = {k: v.cuda() for k, v in inputs.items()}
            
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
#         embeddings.append(batch_embeddings)

#     return np.vstack(embeddings) if len(embeddings) > 0 else np.zeros((0, default_embedding_length))


def chunk_input_ids(input_ids, tokenizer, max_length, stride):
    chunks = []
    for start in range(0, len(input_ids), stride):
        end = start + max_length
        chunk = input_ids[start:end]
        if len(chunk) < max_length:
            # Pad the chunk if necessary
            chunk = torch.cat([
                chunk,
                torch.full((max_length - len(chunk),), tokenizer.pad_token_id, dtype=torch.long)
            ])
        chunks.append(chunk)
        if end >= len(input_ids):
            break
    return chunks


def get_embeddings_chunked(texts, model, tokenizer, batch_size=32, tokenizer_max_length=512, stride=256):
    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    all_chunks = []
    chunk_to_text_idx = []

    # Step 1: Chunk each text and keep track of which text each chunk belongs to
    for idx, text in enumerate(texts):
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False
        )["input_ids"][0]

        chunks = chunk_input_ids(input_ids, tokenizer, tokenizer_max_length, stride)
        all_chunks.extend(chunks)
        chunk_to_text_idx.extend([idx] * len(chunks))

    # Step 2: Batch all chunks for efficient processing
    embeddings_per_text = [[] for _ in texts]
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_input_ids = torch.stack(batch_chunks)
        inputs = {"input_ids": batch_input_ids}

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        batch_text_indices = chunk_to_text_idx[i:i+batch_size]

        for emb, text_idx in zip(batch_embeddings, batch_text_indices):
            embeddings_per_text[text_idx].append(emb)

    # Step 3: Aggregate (mean) chunk embeddings for each original text
    final_embeddings = [np.mean(chunks, axis=0) for chunks in embeddings_per_text]
    return np.vstack(final_embeddings)

def get_notes_data(input: dict, patient_admission_data: pd.DataFrame) -> pd.DataFrame:
    print()
    print('Processing notes data...')
    note_admission_ids = []
    note_ids = []
    note_dts = []
    note_texts = []
    for note in input['notes_input']:
        note_admission_ids.append(note['admission_id'])
        note_ids.append(note['note_id'])
        note_dts.append(note['note_dt'])
        note_texts.append(note['note_text'])
    notes_input = {
        'admission_id': note_admission_ids,
        'note_id': note_ids,
        'note_dt': note_dts,
        'note_text': note_texts,
    }

    notes_data = pd.DataFrame(notes_input)

    notes_data['note_dt'] = pd.to_datetime(notes_data['note_dt'])
    notes_data['note_dt'] = notes_data['note_dt'].dt.tz_localize(None)

    notes_dt = pd.merge(
        patient_admission_data[['admission_id', 'admission_dt']],
        notes_data,
        on='admission_id',
        how='inner',
    )
    assert_unique(notes_dt, 'note_id')
    assert notes_dt['admission_id'].isna().sum() == 0, 'admission_id has NaN values'
    
    return notes_dt

def concatenate_notes(notes_dt: pd.DataFrame) -> pd.DataFrame:
    print('Concatenating notes...')
    notes_dt['hours_since_admission'] = (notes_dt['note_dt'] - notes_dt['admission_dt']).dt.total_seconds() / 3600
    if len(notes_dt) > 0:
        print(f"# of notes entered before admission_dt: {len(notes_dt[notes_dt['hours_since_admission'] < 0])}")
        print(f"{len(notes_dt[notes_dt['hours_since_admission'] < 0])/len(notes_dt):.2%} of notes entered before admission_dt")

    def concatenate_hours_since(row):
        return f"New note {row['hours_since_admission']:.1f} hours since admission: {row['note_text']}"

    notes_concat = notes_dt.copy()
    notes_concat['note_text'] = notes_concat.apply(concatenate_hours_since, axis=1)
    notes_concat.drop(['admission_dt', 'note_dt', 'hours_since_admission'], axis=1, inplace=True)

    notes_concat = notes_concat.groupby('admission_id')['note_text'].apply(lambda x: ' '.join(x)).reset_index()
    assert_unique(notes_concat, 'admission_id')
    
    return notes_concat

def get_note_embeddings(notes_concat: pd.DataFrame) -> pd.DataFrame:
    print('Getting note embeddings...')
    notes_embeddings = get_embeddings_chunked(
        list(notes_concat['note_text']),
        clinicalBERT,
        clinicalBERT_tokenizer,
        batch_size=256,
        tokenizer_max_length=512 # 512 max length is the limit for ClinicalBERT
    )

    assert notes_embeddings.shape[0] == len(notes_concat), "Number of note embeddings does not match input"

    note_features = pd.DataFrame({
        'admission_id': notes_concat['admission_id'],
        'note_embedding': list(notes_embeddings)
    })

    assert_unique(note_features, 'admission_id')
    
    return note_features

def get_prescriptions_data(input: dict) -> pd.DataFrame:
    print()
    print('Processing prescriptions data...')
    prescription_admission_ids = []
    prescription_ids = []
    prescription_start_dates = []
    prescription_names = []
    prescription_doses = []
    for prescription in input['prescriptions_input']:
        prescription_admission_ids.append(prescription['admission_id'])
        prescription_ids.append(prescription['rx_id'])
        prescription_start_dates.append(prescription['rx_start_date'])
        prescription_names.append(prescription['rx_name'])
        prescription_doses.append(prescription['rx_dose'])
    prescriptions_input = {
        'admission_id': prescription_admission_ids,
        'rx_id': prescription_ids,
        'rx_start_date': prescription_start_dates,
        'rx_name': prescription_names,
        'rx_dose': prescription_doses,
    }

    prescriptions_data = pd.DataFrame(prescriptions_input)

    prescriptions_data['rx_start_date'] = pd.to_datetime(prescriptions_data['rx_start_date'])
    prescriptions_data['rx_start_date'] = prescriptions_data['rx_start_date'].dt.tz_localize(None)

    return prescriptions_data

def concatenate_prescriptions(prescriptions_data: pd.DataFrame) -> pd.DataFrame:
    print('Concatenating prescriptions...')
    def concatenate_rx_name_dose(row):
        return f"New prescription {row['rx_name']}: {row['rx_dose']}"

    prescriptions_concat = prescriptions_data.copy()
    prescriptions_concat['prescription_text'] = prescriptions_concat.apply(concatenate_rx_name_dose, axis=1)
    prescriptions_concat.drop(['rx_start_date', 'rx_name', 'rx_dose'], axis=1, inplace=True)

    prescriptions_concat = prescriptions_concat.groupby('admission_id')['prescription_text'].apply(lambda x: ' '.join(x)).reset_index()
    assert_unique(prescriptions_concat, 'admission_id')

    return prescriptions_concat

def get_prescription_embeddings(prescriptions_concat: pd.DataFrame) -> pd.DataFrame:
    print('Getting prescription embeddings...')
    prescriptions_embeddings = get_embeddings_chunked(
        list(prescriptions_concat['prescription_text']),
        clinicalBERT,
        clinicalBERT_tokenizer,
        batch_size=256,
        tokenizer_max_length=512 # 512 max length is the limit for ClinicalBERT
    )

    assert prescriptions_embeddings.shape[0] == len(prescriptions_concat), "Number of prescription embeddings does not match input"

    prescription_features = pd.DataFrame({
        'admission_id': prescriptions_concat['admission_id'],
        'prescription_embedding': list(prescriptions_embeddings)
    })

    assert_unique(prescription_features, 'admission_id')
    
    return prescription_features