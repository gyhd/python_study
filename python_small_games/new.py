
def intToRoman(num):
    s = [''] * 4
    I, V, X, L, C, D, M = 1, 5, 10, 50, 100, 500, 1000
    a = num // M
    b = (num - a * M) // C
    c = (num - a * M - b * C) // X
    d = (num - a * M - b * C - c * X) // I
    s[0] = 'M' * a
    if b < 4:
        s[1] = 'C' * b
    elif b == 4:
        s[1] = 'CD'
    elif b == 5:
        s[1] = 'D'
    elif b == 9:
        s[1] = 'CM'
    else:
        s[1] = 'D' + 'C' * (b - 5)

    if c < 4:
        s[2] = 'X' * c
    elif c == 4:
        s[2] = 'XL'
    elif c == 5:
        s[2] = 'L'
    elif c == 9:
        s[2] = 'XC'
    else:
        s[2] = 'L' + 'X' * (c - 5)

    if d < 4:
        s[3] = 'I' * d
    elif d == 4:
        s[3] = 'IV'
    elif d == 5:
        s[3] = 'V'
    elif d == 9:
        s[3] = 'IX'
    else:
        s[3] = 'V' + 'I' * (d - 5)
    return print("".join(s))


intToRoman(19)


def threeSum(self, nums):
    s = []
    for i in range(len(nums) - 2):
        for j in range(i + 1, len(nums) - 1):
            for k in range(j + 1, len(nums)):
                if nums[i] + nums[j] + nums[k] == 0:
                    s.append([nums[i], nums[j], nums[k]])
    for i in range(len(s) - 1):
        for j in range(i + 1, len(s)):
            if s[i][0] == s[j][2] and s[i][1] == s[j][0] and s[i][2] == s[j][1]:
                s.pop(nums[i + 1])
            elif s[i][0] == s[j][1] and s[i][1] == s[j][2] and s[i][2] == s[j][0]:
                s.pop(nums[i + 1])
            elif s[i][0] == s[j][1] and s[i][1] == s[j][0] and s[i][2] == s[j][2]:
                s.pop(nums[i + 1])
            elif s[i][0] == s[j][0] and s[i][1] == s[j][2] and s[i][2] == s[j][1]:
                s.pop(nums[i + 1])
            elif s[i][::-1] == s[j]:
                s.pop(nums[i + 1])
    return s



def longestCommonPrefix(strs):
    s = []
    for i in range(len(strs[0])):
        k = 0
        while k < len(strs[1]):
            if strs[0][i] == strs[1][k]:
                s.append(strs[0][i])
            k += 1
    s = "".join(s)
    for i in range(len(s) - 1):
        for j in range(2, len(strs)):
            k = i + 1
            while k < len(s):
                if s[i:k] in strs[j]:
                    k += 1
                else:
                    s = s[i: k - 1]
    if s == []:
        return print("""""")
    else:
        return print(''.join(s))

longestCommonPrefix(["ow","flow","flowht"])


