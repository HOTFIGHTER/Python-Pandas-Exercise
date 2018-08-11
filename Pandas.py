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
