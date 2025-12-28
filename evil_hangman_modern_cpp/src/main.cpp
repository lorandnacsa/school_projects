#include "game.h"
#include "tui.h"
#include <bits/stdc++.h>
#include <string>
#include <iostream>

int main(int argc, char** argv){
    std::ios::sync_with_stdio(false);
    std::cin.tie(&std::cout);

    std::string dict_path;
    if(argc >= 2) dict_path = argv[1];

    int word_length = 0;
    std::cout << "Enter the word length: ";
    while(!(std::cin >> word_length) || word_length <= 0){
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Try again: ";
    }

    int max_mistakes = 8;
    std::cout << "Allowed mistakes (default 8). Press Enter to accept default: ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string line;
    std::getline(std::cin, line);
    if(!line.empty()){
        try {
            int v = std::stoi(line);
            if(v > 0) max_mistakes = v;
        } catch(...) {}
    }


    Game game(dict_path, word_length, max_mistakes);
    game.start();




}
