from typing import List


# 2 строки s и t
# анограмма - слово или фраза может при перестановкой букв должно получиться другое слово
def majorityElement(nums: List[int]):
    count_dict = {}
    for i in nums:
        if i not in count_dict:
            count_dict.update({i: 1})
        else:
            count_dict[i] += 1
    max_result = max(count_dict.values())
    for key, value in count_dict.items():
        if value == max_result:
            return key


if __name__ == '__main__':
    # cat and  dog
    print(f"Case 1 --- {majorityElement([3, 2, 3])}")
    print(f"Case 2 --- {majorityElement([2, 2, 1, 1, 1, 2, 2])}")
    print(f"Case 3 --- {majorityElement([3, 3, 4])}")
