# coding=utf-8
import pandas as pd
import numpy as np
print(pd.__version__)
arr=[0,1,2,3,4,5]
s1=pd.Series(arr)
#带有index赋值的series
n=np.random.random(5)
index=['a','b','c','d','e']
s2=pd.Series(n,index=index)
#从字典创建
n={'a':1,'b':2,'c':3,'d':4,'e':5}
s3=pd.Series(n)
#修改索引
s3.index=['A','B','C','D','E']
#纵向合并
s4=s3.append(s1)
print(s4)
#删除索引元素
s4.drop('E')
s4['A']=3
print(s4['A'])
#切片显示前三个数
print(s4[:3])
print(s3.add(s2))
print(s3.mul(s2))
#求series的中位数
s3.median()
#DataFrame
dates=pd.date_range('today',periods=6) # 定义时间序列作为 index
num_arr=np.random.randn(6,4) # 传入 numpy 随机数组
columns=['A','B','C','D'] # 将列表作为列名
df1=pd.DataFrame(num_arr,index=dates,columns=columns)
print(df1)
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df2 = pd.DataFrame(data, index=labels)
print(df2)
#查看dataframe的数据类型
print(df2.dtypes)
#查看dataframe前五行
df2.head(5)
df2.tail(2)
print(df2.index)
print(df2.columns)
print(df2.values)
#查看统计数据
print(df2.describe())
df2.T
#排序
df2.sort_values(by='age')
print(df2.iloc[1:3])
#判断为空
df2.isnull()
#添加列数据
num=pd.Series([0,1,2,3,4,5,6,7,8,9],index=df2.index)
df2['No.']=num # 添加以 'No.' 为列名的新数据列
print(df2)
#对下标值进行修改
df2.iat[1,0]=2
#对缺少值填充
df3=df2.copy()
df3.fillna(value=3)
#删除存在缺少值的行
df3.dropna(how='any')
# DataFrame 按指定列对齐
left = pd.DataFrame({'key': ['foo1', 'foo2'], 'one': [1, 2]})
right = pd.DataFrame({'key': ['foo2', 'foo3'], 'two': [4, 5]})
print(left)
print(right)
# 按照 key 列对齐连接，只存在 foo2 相同，所以最后变成一行
pd.merge(left, right, on='key')
df3.to_csv('animal.csv')
print("写入成功.")
df_animal=pd.read_csv('animal.csv')
print(df_animal)
df3.to_excel('animal.xlsx', sheet_name='Sheet1')
print("写入成功.")
pd.read_excel('animal.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
#进阶
#创建2018以天为单位值为随机数的series
dti=pd.date_range(start='2018-01-01',end='2018-12-31',freq='D')
s=pd.Series(np.random.rand(len(dti)),index=dti)
print(s[s.index.weekday==2].sum())
#统计每个月的平均值
s.resample('M').mean()
s=pd.date_range('today',periods=100,freq='S')
ts=pd.Series(np.random.randint(0,500,len(s)),index=s)
print('sum')
print(ts.resample('Min').sum())
ts_utc=ts.tz_localize('UTC')#世界标准时间
ts_utc.tz_convert('Asia/Shanghai')
ps=ts.to_period()#按间隔划分
ps.to_timestamp()#按开始时间划分
#多重索引
letters = ['A', 'B', 'C']
numbers = list(range(10))
mi = pd.MultiIndex.from_product([letters, numbers]) # 设置多重索引
s = pd.Series(np.random.rand(30), index=mi) # 随机数
#多重索引查询
s.loc[:, [1, 3, 6]]
#多重索引切片
s.loc[pd.IndexSlice[:'B', 5:]]
#DataFrame的多重索引
frame = pd.DataFrame(np.arange(12).reshape(6, 2),
                     index=[list('AAABBB'), list('123123')],
                     columns=['hello', 'shiyanlou'])
frame.index.names=['first','second']
frame.groupby('first').sum()
#DataFrame的条件查找
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(data, index=labels)
df[df['age'] > 3]
#按行列切片索引
df.iloc[2:4, 1:3]
#多条件查找
df[(df['animal'] == 'cat') & (df['age'] < 3)]
#关键词查找
df3[df3['animal'].isin(['cat', 'dog'])]
#多条件排序
print(df.sort_values(by=['age', 'visits'], ascending=[False, True]))
#多值替换
df['priority'].map({'yes': True, 'no': False})
#分组求和
df.groupby('animal').sum()
#使用拼接多个DataFrame
temp_df1 = pd.DataFrame(np.random.randn(5, 4)) # 生成由随机数组成的 DataFrame 1
temp_df2 = pd.DataFrame(np.random.randn(5, 4)) # 生成由随机数组成的 DataFrame 2
temp_df3 = pd.DataFrame(np.random.randn(5, 4)) # 生成由随机数组成的 DataFrame 3
print(temp_df1)
print(temp_df2)
print(temp_df3)
pieces = [temp_df1,temp_df2,temp_df3]
print(pd.concat(pieces))
df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
print(df)
print(df.sum())
df.sum().idxmin()  # idxmax(), idxmin() 为 Series 函数返回最大最小值的索引值
#DataFrame 中每个元素减去每一行的平均值
df = pd.DataFrame(np.random.random(size=(5, 3)))
print(df)
df.sub(df.mean(axis=1), axis=0)
df = pd.DataFrame({'A': list('aaabbcaabcccbbc'),
                   'B': [12, 345, 3, 1, 45, 14, 4, 52, 54, 23, 235, 21, 57, 3, 87]})
print(df.groupby('A')['B'].nlargest(3).sum(level=0))
#透视表的创建
#新建表将 A, B, C 列作为索引进行聚合。
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})
print(df)
pd.pivot_table(df, index=['A', 'B'])
pd.pivot_table(df,values=['D'],index=['A', 'B'])
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df["grade"] = df["raw_grade"].astype("category")
print(df)
df["grade"].cat.categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df.sort_values(by="grade")
df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
                               'Budapest_PaRis', 'Brussels_londOn'],
                   'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
                   'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
                               '12. Air France', '"Swiss Air"']})
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)
#数据列的差分
temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']
temp['From'] = temp['From'].str.capitalize()
temp['To'] = temp['To'].str.capitalize()
df = df.drop('From_To', axis=1)
df = df.join(temp)
print(df)
df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip()
#数据去重
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
df.loc[df['A'].shift() != df['A']]
#数据归一化
def normalization(df):
    numerator=df.sub(df.min())
    denominator=(df.max()).sub(df.min())
    Y=numerator.div(denominator)
    return Y
df = pd.DataFrame(np.random.random(size=(5, 3)))
print(df)
normalization(df)
#可视化
import matplotlib.pyplot as plt
ts = pd.Series(np.random.randn(100), index=pd.date_range('today', periods=100))
ts = ts.cumsum()
ts.plot()

df = pd.DataFrame(np.random.randn(100, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
df.plot()

df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})
df = df.cumsum()
df.plot.scatter("xs","ys",color='red',marker="*")

df = pd.DataFrame({"revenue": [57, 68, 63, 71, 72, 90, 80, 62, 59, 51, 47, 52],
                   "advertising": [2.1, 1.9, 2.7, 3.0, 3.6, 3.2, 2.7, 2.4, 1.8, 1.6, 1.3, 1.9],
                   "month": range(12)
                   })
ax = df.plot.bar("month", "revenue", color="yellow")
df.plot("month", "advertising", secondary_y=True, ax=ax)