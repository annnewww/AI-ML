def longestUniqueWord(sentence):
    words = sentence.split()
    longest_word = ""
    for word in words:
        count = 0
        for w in words:
            if w == word:
                count += 1

        if count == 1 and len(word) > len(longest_word):
            longest_word = word

    return longest_word


def testLongestUniqueWord(sentence, expected):
    result = longestUniqueWord(sentence)

    if result == expected:
        print(f'longestUniqueWord("{sentence}")... Ok.')
    else:
        print(f'longestUniqueWord("{sentence}")... Error!')
        print(f"     expected: {expected}")
        print(f"     got:      {result}")


def main():
    testLongestUniqueWord(
        "cat elephant dog elephant tiger",
        "tiger"
    )

    testLongestUniqueWord(
        "apple banana apple orange banana grape",
        "orange"
    )

    testLongestUniqueWord(
        "java python java c cpp python",
        "cpp"
    )

    testLongestUniqueWord(
        "hello hello hello",
        ""
    )

    testLongestUniqueWord(
        "one two three four",
        "three"
    )

    testLongestUniqueWord(
        "red blue green blue red yellow",
        "yellow"
    )

    testLongestUniqueWord(
        "a bb ccc dddd ccc bb a",
        "dddd"
    )


if __name__ == "__main__":
    main()