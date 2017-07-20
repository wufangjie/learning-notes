import re
from datetime import date, timedelta
import os


holidays = {}
holidays[2017] = '''
元旦：1月1日放假，1月2日（星期一）补休。
春节：1月27日至2月2日放假调休，共7天。1月22日（星期日）、2月4日（星期六）上班。
清明节：4月2日至4日放假调休，共3天。4月1日（星期六）上班。
劳动节：5月1日放假，与周末连休。
端午节：5月28日至30日放假调休，共3天。5月27日（星期六）上班。
中秋节、国庆节：10月1日至8日放假调休，共8天。9月30日（星期六）上班。
'''


fixed = '''
(setq calendar-holidays  ; 直接修改这个变量, 可以屏蔽多余节日
      '((holiday-fixed  2 14 "情人节")
        (holiday-fixed  3  8 "妇女节")
        (holiday-fixed  6  1 "儿童节")
        (holiday-fixed 12 24 "平安夜")
        (holiday-fixed 12 25 "圣诞节")
        {}
        (holiday-chinese 1 1 "大年初一")
        (holiday-chinese 7 7 "七夕")
	(holiday-chinese 8 15 "中秋")
       ))
(provide 'my-holidays)
'''


reg_mdmd = re.compile(r'(\d+)月(\d+)日至(\d+)月(\d+)日')
reg_mdd = re.compile(r'(\d+)月(\d+)日至(\d+)日')
reg_md = re.compile(r'(\d+)月(\d+)日')

year = date.today().year
sep = '\n        '
templates = ['(holiday-fixed {:>2} {:>2} "{}")',
             '{:0>2}/{:0>2}/' + str(year) + '  上班  --补{}放假']
outputs = [[], []]
for row in holidays[year].split('\n')[1:-1]:
    name, arrange = row.split('：')
    for i, same in enumerate(arrange[:-1].split('。')):
        temp = (reg_mdmd.findall(same) or reg_mdd.findall(same)
                or reg_md.findall(same))
        if len(temp[0]) == 4:
            m1, d1, m2, d2 = map(int, temp[0])
        elif len(temp[0]) == 3:
            m1, d1, d2 = map(int, temp[0])
            m2 = m1
        else:
            for m, d in temp:
                outputs[i].append(templates[i].format(m, d, name))
            break

        d = date(year, m1, d1)
        d2 = date(year, m2, d2)
        while d <= d2:
            outputs[i].append(templates[i].format(d.month, d.day, name))
            d += timedelta(1)


for m, d, name, i in []: # custom date
    outputs[i].append(templates[i].format(m, d, name))

output_holidays = fixed.format(sep.join(outputs[0]))
print(output_holidays)


if __name__ == '__main__':
    _path = os.path.join(os.getenv('HOME'), '.emacs.d/')
    with open(os.path.join(_path, 'my-holidays.el'), 'wt') as f:
        f.write(output_holidays)
    with open(os.path.join(_path, 'my-diary.txt'), 'wt') as f:
        f.write('\n'.join(outputs[1]))
