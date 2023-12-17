import math
import os
import string
import unicodedata

def extract_president_names(file_names):
    president_names = set()

    for file_name in file_names:
        if file_name.startswith("Nomination_") and file_name.endswith(".txt"):
            parts = file_name[len("Nomination_"):-len(".txt")].split('_')
            if len(parts) > 0:
                president_names.add(parts[0])

    return list(president_names)

def associate_first_names(president_names):
    first_names = {
        "Chirac": "Jacques",
        "Giscard d'Estaing": "Valéry",
        "Mitterrand": "François",
        "Macron": "Emmanuel",
        "Sarkozy": "Nicolas",
        "Hollande": "Francois"
    }

    president_first_names = {}

    for president in president_names:
        if president in first_names:
            president_first_names[president] = first_names[president]

    return president_first_names

def display_president_names(president_last_names, president_first_names):
    print("List of Presidents:")
    for last_name in president_last_names:
        first_name = president_first_names.get(last_name, "Unknown")
        print(f"{first_name} {last_name}")

def convert_to_lowercase_and_remove_punctuation(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read().lower()

        # Split the content into lines
        lines = content.split('\n')

        # Process each line separately
        processed_lines = [remove_punctuation(line) for line in lines]

    with open(output_path, 'w', encoding='utf-8') as file:
        # Join the processed lines back into a single string with newlines
        processed_content = '\n'.join(processed_lines)
        file.write(processed_content)



def process_speeches(speeches_dir, cleaned_dir):
    # Ensure the "cleaned" directory exists
    os.makedirs(cleaned_dir, exist_ok=True)

    # Process each file in the "speeches" directory
    for filename in os.listdir(speeches_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(speeches_dir, filename)
            output_path = os.path.join(cleaned_dir, filename)

            convert_to_lowercase_and_remove_punctuation(input_path, output_path)

            # Add an extra newline at the end of each processed file to separate paragraphs
            with open(output_path, 'a', encoding='utf-8') as file:
                file.write('\n')




def remove_punctuation(text):
    # Unicode character categories to keep (Letters, Numbers, Spaces)
    valid_categories = {'Ll', 'Lu', 'Lt', 'Lm', 'Lo', 'Nd', 'Zs'}

    # Replace dashes and apostrophes with spaces
    text = text.replace('-', ' ').replace("'","")

    # Replace other punctuation with spaces
    text = ''.join(' ' if char in string.punctuation else char for char in text)

    # Remove diacritics (accents)
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) in valid_categories)

    return text


def count_word_occurrences(text):
    word_counts = {}
    words = text.split()
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts

def calculate_idf(cleaned_dir):
    total_documents = len(os.listdir(cleaned_dir))  # Initialize with the total number of documents
    word_document_count = {}

    for filename in os.listdir(cleaned_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(cleaned_dir, filename)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Split the text into words and create a set for unique words in the document
            unique_words_in_document = set(content.split())

            # Update word_document_count for each unique word in the document
            for word in unique_words_in_document:
                if word in word_document_count:
                    word_document_count[word] += 1
                else:
                    word_document_count[word] = 1

    # Calculate IDF for each word with smoothing
    idf_scores = {}
    smoothing_factor = 1e-10  # Small smoothing factor
    for word, document_count in word_document_count.items():
        idf_scores[word] = math.log10((total_documents + smoothing_factor) / (1 + document_count))

    return idf_scores


def calculate_tfidf_matrix(cleaned_dir):
    idf_scores = calculate_idf(cleaned_dir)
    files = [f for f in os.listdir(cleaned_dir) if f.endswith('.txt')]

    tfidf_matrix = [['Word'] + files]

    all_word_counts = {}  # Store word counts for all documents

    for file in files:
        file_path = os.path.join(cleaned_dir, file)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        word_counts = count_word_occurrences(content)
        all_word_counts[file] = word_counts

    unique_words = set(word for counts in all_word_counts.values() for word in counts.keys())

    for word in unique_words:
        tfidf_row = [word]

        for file in files:
            word_count = all_word_counts[file].get(word, 0)
            idf_score = idf_scores.get(word, 0)
            tfidf_score = round(word_count * idf_score, 2)  # Round to the second decimal place

            tfidf_row.append(tfidf_score)

        tfidf_matrix.append(tfidf_row)

    return tfidf_matrix

def is_unimportant(word, tfidf_matrix):
    # Check if the word's TF-IDF score is 0 in all files
    for row in tfidf_matrix[1:]:
        if row[0] == word:
            if all(val == 0 for val in row[1:]):
                return True  # Word has TF-IDF = 0 in all files
            else:
                return False  # Word has non-zero TF-IDF in at least one file
    return False

def tokenize_question(question_text):
    cleaned_question = remove_punctuation(question_text.lower())
    question_words = cleaned_question.split()
    question_words = [word for word in question_words if word]
    return question_words


def find_matching_terms(question_words, tfidf_matrix):
    unique_corpus_words = [row[0] for row in tfidf_matrix[1:]]  # Extract unique words from the first column of the TF-IDF matrix
    matching_terms = set(question_words) & set(unique_corpus_words)
    return list(matching_terms)

def calculate_tfidf_vector(question_words, idf_scores, tfidf_matrix):
    tf_vector = {word: question_words.count(word) / len(question_words) for word in question_words}
    tfidf_vector = {word: tf * idf_scores.get(word, 0) for word, tf in tf_vector.items()}

    # Ensure the vector has the same dimension as the TF-IDF matrix
    unique_corpus_words = [row[0] for row in tfidf_matrix[1:]]
    tfidf_vector = {word: tfidf_vector.get(word, 0) for word in unique_corpus_words}

    return tfidf_vector

##vector_a = list(tfidf_vector.values())
##vector_b = [tfidf_matrix[i][1:] for i in range(1, len(tfidf_matrix))]
def dot_product(vector_a, vector_b):
    return sum(a * b for a, b in zip(vector_a, vector_b))
def vector_norm(vector):
    return math.sqrt(sum(v**2 for v in vector))

def cosine_similarity(vector_a, vector_b):
    if isinstance(vector_a, dict):
        vector_a = list(vector_a.values())
    if isinstance(vector_b, dict):
        vector_b = list(vector_b.values())

    dot_product_ab = dot_product(vector_a, vector_b)
    norm_a = vector_norm(vector_a)
    norm_b = vector_norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0  # Handle division by zero

    similarity_score = dot_product_ab / (norm_a * norm_b)
    return similarity_score
def find_most_relevant_document(tfidf_matrix, question_vector, file_names):
    max_similarity = -1
    most_relevant_document = None

    for doc_vector, file_name in zip(tfidf_matrix[1:], file_names):
        similarity = cosine_similarity(question_vector, doc_vector[1:])
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_document = file_name

    return most_relevant_document

import os

def convert_cleaned_to_speeches(cleaned_file):
    if not cleaned_file.endswith('.txt'):
        raise ValueError("Input file must have a '.txt' extension")

    cleaned_dir = 'cleaned'
    speeches_dir = 'speeches'

    cleaned_path = os.path.join(cleaned_dir, cleaned_file)
    speeches_file = cleaned_file.replace('.txt', '_speech.txt')
    speeches_path = os.path.join(speeches_dir, speeches_file)

    # Read content from the cleaned file
    with open(cleaned_path, 'r', encoding='utf-8') as cleaned_file:
        cleaned_content = cleaned_file.read()

    # Perform any additional processing if needed before saving to speeches
    speeches_content = cleaned_content  # For example, you might want to do more processing here

    # Save content to the speeches directory
    with open(speeches_path, 'w', encoding='utf-8') as speeches_file:
        speeches_file.write(speeches_content)

    return speeches_path
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_menu():
    print("===== Main Menu =====")
    print("1. Extract President Names")
    print("2. Process Speeches")
    print("3. Search for 'climate' or 'ecology'")
    print("4. Test Question Functions")
    print("5. Check Most Relevant Document")
    print("6. Exit")
    print("=====================")
def extract_president_names_menu():
    file_names = ["Nomination_Chirac.txt", "Nomination_Macron.txt", "Nomination_Mitterrand_2.txt",
                  "Nomination_Giscard_d'Estaing.txt", "Nomination_Sarkozy.txt", "Nomination_Hollande.txt"]

    president_names = extract_president_names(file_names)
    president_first_names = associate_first_names(president_names)
    display_president_names(president_names, president_first_names)

def process_speeches_menu():
    speeches_dir = 'speeches'
    cleaned_dir = 'cleaned'
    process_speeches(speeches_dir, cleaned_dir)
    print("Speeches processed and cleaned.")

def search_climate_ecology_menu():
    speeches_dir = 'speeches'
    if os.path.exists(speeches_dir) and os.path.isdir(speeches_dir):
        found_files = []
        for filename in os.listdir(speeches_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(speeches_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if 'climate' in content.lower() or 'ecology' in content.lower():
                        found_files.append(filename)

        if found_files:
            print("Files containing 'climate' or 'ecology':")
            for file in found_files:
                print(file)
        else:
            print("No files found containing 'climate' or 'ecology'.")
    else:
        print(f"Directory '{speeches_dir}' does not exist or is not a directory.")

def test_question_functions():
    question_text = "Peux-tu me dire comment une nation peut-elle prendre soin du climat ?"

    # Tokenize the question
    question_words = tokenize_question(question_text)
    print("Tokenized Question Words:", question_words)

    # Assuming you already have the TF-IDF matrix and IDF scores
    tfidf_matrix = calculate_tfidf_matrix('cleaned')  # Adjust the directory as needed
    idf_scores = calculate_idf('cleaned')  # Adjust the directory as needed

    # Find matching terms in the TF-IDF matrix
    matching_terms = find_matching_terms(question_words, tfidf_matrix)
    print("Matching Terms in the TF-IDF Matrix:", matching_terms)

    # Calculate TF-IDF vector for the question
    tfidf_vector = calculate_tfidf_vector(question_words, idf_scores, tfidf_matrix)
    print("TF-IDF Vector for the Question:", tfidf_vector)

def find_files_containing_highest_tfidf_word(tfidf_matrix, idf_scores, speeches_dir, question_text):
    # Tokenize the question
    question_words = tokenize_question(question_text)

    # Calculate TF-IDF vector for the question
    tfidf_vector_question = calculate_tfidf_vector(question_words, idf_scores, tfidf_matrix)

    # Find the word with the highest TF-IDF score in the question
    max_tfidf_word_question = max(tfidf_vector_question, key=tfidf_vector_question.get)

    files = [f for f in os.listdir(speeches_dir) if f.endswith('.txt')]

    files_containing_word = []

    for file in files:
        file_path = os.path.join(speeches_dir, file)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Calculate TF-IDF vector for the document
        tfidf_vector_doc = calculate_tfidf_vector(tokenize_question(content), idf_scores, tfidf_matrix)

        # Check if the word with the highest TF-IDF score in the question is present in the document
        if max_tfidf_word_question.lower() in tfidf_vector_doc:
            files_containing_word.append(file)

    return files_containing_word



def find_response_for_question(tfidf_matrix, idf_scores, speeches_dir, question_text):
    # Tokenize the question
    question_words = tokenize_question(question_text)

    # Calculate TF-IDF vector for the question
    tfidf_vector_question = calculate_tfidf_vector(question_words, idf_scores, tfidf_matrix)

    # Find the word with the highest TF-IDF score in the question
    max_tfidf_word_question = max(tfidf_vector_question, key=tfidf_vector_question.get)

    # Find the index of the file where the word with the highest TF-IDF score occurs
    word_index = [word[0] for word in tfidf_matrix].index(max_tfidf_word_question)
    file_index = tfidf_matrix[word_index].index(max(tfidf_matrix[word_index][1:])) - 1

    # Get the file name from the TF-IDF matrix
    file_name = tfidf_matrix[0][file_index + 1]

    # Read the content of the selected file
    selected_file_path = os.path.join(speeches_dir, file_name)
    with open(selected_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Find the first occurrence of the word with the highest TF-IDF score in the selected file
    lines = content.split('\n')
    response_lines = []
    for line_num, line in enumerate(lines):
        if max_tfidf_word_question.lower() in tokenize_question(line):
            start_line = max(0, line_num - 1)  # Get the line before the occurrence
            end_line = min(len(lines), line_num + 2)  # Get the line after the occurrence (total 2 lines)
            response_lines = lines[start_line:end_line]
            break

    # Fetch question starters
    question_starters_dict = questions_starters()

    # Identify if the question starts with a known question form
    question_start = None
    for question_form, intro_phrase in question_starters_dict.items():
        if question_text.startswith(question_form):
            question_start = intro_phrase
            break

    # Construct the response with an introductory phrase
    if question_start:
        response_lines.insert(0, question_start)

    return '\n'.join(response_lines)


def questions_starters():
    questions_starters = {
        "Comment": "Après réflexion,",
        "Pourquoi": "Car,",
        "Es-ce que": "En effet,",
        "Peux-tu": "Bien sûr,",
        "Pouvez-vous": "Bien sûr,"
    }
    return questions_starters

def find_president_by_word(tfidf_matrix, speeches_dir):
    word = input("Enter a word to search in speeches: ")
    word = word.lower()  # Convert the input word to lowercase for case-insensitive matching

    max_occurrences = 0
    president_with_most_occurrences = None

    # Loop through each speech file
    for filename in os.listdir(speeches_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(speeches_dir, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Count occurrences of the word in the content
            word_count = content.lower().count(word)

            # Check if the current speech has more occurrences of the word than the previous ones
            if word_count > max_occurrences:
                max_occurrences = word_count
                president_with_most_occurrences = filename.split('_')[1].split('.')[0]  # Extract president's name

    if president_with_most_occurrences:
        print(f"The president who spoke the most about '{word}' is: {president_with_most_occurrences}")
    else:
        print(f"No president spoke about '{word}' in their speeches.")