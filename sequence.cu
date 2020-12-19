#include <iostream>
#include <random>

int randWithBitMasking() {

    return (random() >> 20) & INT32_MAX;
}

void generateSequences(int *sequences, const int count) {
    for (int i = 0; i < count * SEQUENCE_LEN; i++) {
        sequences[i] = randWithBitMasking();

    }
}

void printPairs(const bool *isPair, const int *seqs) {
    int counter = 0;
    for (int i = 0; i < SEQUENCE_COUNT * SEQUENCE_COUNT; i++) {
        if (isPair[i]) {
            counter++;
#if PRINT_PAIRS
            unsigned int x = i / SEQUENCE_COUNT;
            unsigned int y = i - x * SEQUENCE_COUNT;
            printf("-------%d-%d--------\n", y, x);
            for (int j = 0; j < SEQUENCE_LEN; j++) printf("%d ", seqs[y * SEQUENCE_LEN + j]);
            printf("\n");
            for (int j = 0; j < SEQUENCE_LEN; j++) printf("%d ", seqs[x * SEQUENCE_LEN + j]);
            printf("\n----------------\n\n");
#endif
        }
    }
    std::cout << "Pairs of hamming one count = " << counter << std::endl;

}

void printGeneratedSeqs(const int *seqs) {
    for (int i = 0; i < SEQUENCE_COUNT; i++) {

        for (int j = 0; j < SEQUENCE_LEN; j++) {
            std::cout << seqs[i * SEQUENCE_LEN + j] << " ";
        }
        std::cout << std::endl;
    }
}

void resetPairs(bool *pairs, int count) {
    for (int i = 0; i < count * count; i++)
        pairs[i] = false;
}

void print_timediff(const char *prefix, const timespec &start, const timespec &end) {
    double milliseconds = end.tv_nsec >= start.tv_nsec
                          ? (end.tv_nsec - start.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec) * 1e3
                          : (start.tv_nsec - end.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec - 1) * 1e3;
    printf("%s: %lf milliseconds\n", prefix, milliseconds);
}

void print_timediff(const char* prefix, const struct timespec& start, const
struct timespec& end);