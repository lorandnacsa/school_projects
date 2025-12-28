#include <iostream>
#include "tui.h"
#include "game.h"




Tui::Tui(Game& g) : game(g) {}

void Tui::clear_screen() {
    std::cout << "\033[2J\033[H"; // ANSI clear
}

void Tui::draw_hangman() {
    int wrong = game.get_wrong_guesses().size();

    const char* stages[] = {
        "  +---+\n  |   |\n      |\n      |\n      |\n      |\n=========\n",
        "  +---+\n  |   |\n  O   |\n      |\n      |\n      |\n=========\n",
        "  +---+\n  |   |\n  O   |\n  |   |\n      |\n      |\n=========\n",
        "  +---+\n  |   |\n  O   |\n /|   |\n      |\n      |\n=========\n",
        "  +---+\n  |   |\n  O   |\n /|\\  |\n      |\n      |\n=========\n",
        "  +---+\n  |   |\n  O   |\n /|\\  |\n /    |\n      |\n=========\n",
        "  +---+\n  |   |\n  O   |\n /|\\  |\n / \\  |\n      |\n=========\n"
    };

    int total_stages = sizeof(stages)/sizeof(stages[0]);

    int index = (wrong * (total_stages - 1)) / game.get_max_wrong();
    if(index < 0) index = 0;
    if(index >= total_stages) index = total_stages - 1;

    std::cout << stages[index];
}

void Tui::print_guesses() {
    auto wrong = game.get_wrong_guesses();
    auto all   = game.get_all_guesses();

    std::cout << "\nWrong guesses: ";
    for (char c : wrong) std::cout << c << " ";
    if (wrong.empty()) std::cout << "-";

    std::cout << "\nAll guesses:   ";
    for (char c : all) std::cout << c << " ";
    if (all.empty()) std::cout << "-";
    std::cout << "\n";
}

void Tui::run_terminal_ui() {
    while (!game.is_over()) {
        clear_screen();

        draw_hangman();

        std::cout << "\nWORD: " << game.get_display_word() << "\n";
        print_guesses();

        std::cout << "\nEnter a letter: ";
        std::string input;
        std::getline(std::cin, input);

        if (input.empty()) continue;

        char c = static_cast<char>(std::tolower(static_cast<unsigned char>(input[0])));

        if (!std::isalpha(static_cast<unsigned char>(c))) {
            std::cout << "Please enter a valid letter!\n(Press Enter...)";
            std::getline(std::cin, input);
            continue;
        }

        game.guess(c);
    }

    clear_screen();

    if (game.is_won()) {
        std::cout << "You won! The word was: " << game.get_display_word() << "\n";
    } else {
        std::cout << "You lost!\nThe correct word was: " << game.reveal_solution() << "\n";
        draw_hangman();
    }

    std::cout << "\nThanks for playing!\n";
}
