from rag import RAG


def main():
    print("=" * 50)
    print("  Mon Premier RAG")
    print("=" * 50)
    print("Tapez 'quit' pour quitter.\n")

    rag = RAG()

    while True:
        question = input("\nQuestion : ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        response = rag.answer_question(question)
        print(f"\nRéponse : {response}")


if __name__ == "__main__":
    main()
