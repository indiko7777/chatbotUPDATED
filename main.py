
from functions import *
import os
def display_tfidf_matrix(tfidf_matrix):
    print("TF-IDF Matrix:")
    for row in tfidf_matrix:
        print("\t".join(str(item) for item in row))

def main():
    tfidf_matrix = calculate_tfidf_matrix('cleaned')
    idf_scores = calculate_idf('cleaned')

    while True:
        print("===== Main Menu =====")
        print("1. Extract President Names")
        print("2. Process Speeches")
        print("3. Search president who speak the most about a word'")
        print("4. Test Question Functions")
        print("5. Check Most Relevant Document")
        print("6. Ask a Question to the chat-bot")
        print("7. Display TF-IDF Matrix")
        print("8. Exit")
        print("=====================")
        choice = input("Enter your choice (1-8): ")

        if choice == '1':
            extract_president_names_menu()
        elif choice == '2':
            process_speeches_menu()
        elif choice == '3':  # Modify this section to replace the search option
            find_president_by_word(tfidf_matrix, 'speeches')
        elif choice == '4':
            test_question_functions()
        elif choice == '5':
            check_most_relevant_document_menu(tfidf_matrix, idf_scores)
        elif choice == '6':
            speeches_dir = 'speeches'  # Modify this according to your directory structure
            question = input("Enter your question: ")
            print("Question:", question)
            response_lines = find_response_for_question(tfidf_matrix, idf_scores, speeches_dir, question)
            print("Response for the question:")
            print(response_lines)
        elif choice == '7':
            display_tfidf_matrix(tfidf_matrix)
        elif choice == '8':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")

if __name__ == "__main__":
    main()