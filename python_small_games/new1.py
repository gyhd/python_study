

def threeSum(nums):
    s = []
    for i in range(len(nums) - 2):
        for j in range(i + 1, len(nums) - 1):
            for k in range(j + 1, len(nums)):
                if nums[i] + nums[j] + nums[k] == 0:
                    s.append([nums[i], nums[j], nums[k]])
    s.sort()

    for i in range(len(s)-5):
        if s[i] == s[i+1]:
            s.pop(i)
    #     s.pop(i) if s[i] == s[i+1]
        # j = 1
        # while j < len(s):
        #     if s[i] == s[j]:
        #         s.pop(j)
        #     j += 1
    # for i in range(0, len(s)-1):
    #     for j in range(i + 1, len(s)):
    #         if s[i][0] == s[j][2] and s[i][1] == s[j][0] and s[i][2] == s[j][1]:
    #             s.pop(j)
    #         if s[i][0] == s[j][1] and s[i][1] == s[j][2] and s[i][2] == s[j][0]:
    #             s.pop(j)
    #         if s[i][0] == s[j][1] and s[i][1] == s[j][0] and s[i][2] == s[j][2]:
    #             s.pop(j)
    #         if s[i][0] == s[j][0] and s[i][1] == s[j][2] and s[i][2] == s[j][1]:
    #             s.pop(j)
    #         if s[i][::-1] == s[j]:
    #             s.pop(j)
    return print(s)


threeSum([-1, 0, 1, 2, -1, -4, 1, 2, -2, 0, 3])

