from capture_faces import capture_faces
from pca_module import train_model
from recognition import recognize_live


def main():
    model = None

    while True:
        print("\n==============================")
        print("        EIGENVISION")
        print("     PCA Face Recognition")
        print("==============================")
        print("1. Capture faces")
        print("2. Train PCA model")
        print("3. Show eigenfaces")
        print("4. Start face recognition")
        print("5. Exit")

        choice = input("\nChoose an option: ").strip()

        try:
            if choice == "1":
                capture_faces()

            elif choice == "2":
                model = train_model()

            elif choice == "3":
                if model is None:
                    model = train_model()

                from visualization import show_eigenfaces
                show_eigenfaces(model)

            elif choice == "4":
                if model is None:
                    model = train_model()

                recognize_live(model)

            elif choice == "5":
                print("Goodbye!")
                break

            else:
                print("Please choose 1 to 5.")

        except KeyboardInterrupt:
            print("\nStopped.")
        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
