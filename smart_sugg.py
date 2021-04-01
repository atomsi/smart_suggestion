import pandas as pd
import numpy as np
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder 

"""
# 智能推荐系统
- 以“客户需求”为导向， 提高商品的活力、挖掘消费者的购买力、促进最大化销售
- 基于关联规则的购物篮推荐,其建模理念为：物品被同时购买的模式反映了客户的需求模式
- 适用场景 ：无需个性化的场景；有销售记录的产品，向老客户推荐；套餐设计与产品摆放。
"""

DataFile = "./data/shop_data.csv"

@st.cache
def load_data(file):
    data = pd.read_csv(file, encoding='UTF-8')
    return data;


def data_status(df):
    st.subheader ('原始数据')
    # 检查重复值
    st.write('总记录:',df.shape[0],'条, 共有商品:', df['Model'].nunique(),' 共有客户:', df['CustomerID'].nunique())
    st.write(df[:3])
    # 最畅销的 5 种商品
    ## 往 return eset_index 中添加 name 参数可快速重命名列名
    # st.write('Top 5 Goods:')
    # grouped = df.groupby('Model')['Model'].count().reset_index(name='count')
    # top = grouped.sort_values(by='count', ascending=False).head(5)
    # st.write(top)
 
def encode_units(x):
    if x <= 0:
        return 0
    if x > 0:
        return 1

def suggestion_model(df):
    st.sidebar.subheader("模型参数")
    min_support_s = st.sidebar.slider('最小支持度阈值:',0.01,0.2,0.005,0.01)
    metric_s = st.sidebar.selectbox('度量方法:',('lift','support','confidence'))
    min_threshold_s = st.sidebar.slider('度量阈值:',0.0,1.0,0.8,0.1)
    st.subheader("构建模型")
    basket = df.pivot_table(columns = "Model",index="CustomerID",
                            values="LineNumber",aggfunc=np.sum).fillna(0)
    basket_sets = basket.applymap(encode_units)
    #basket_sets.drop('LineNumber', inplace=True, axis=1)
    st.write(basket_sets.head(5))
    frequent_itemsets = apriori(basket_sets, min_support=min_support_s, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=metric_s, min_threshold=min_threshold_s)
    st.write (frequent_itemsets[:5])

    st.sidebar.subheader("推荐结果筛选参数")
    lift = st.sidebar.slider('提升度筛选:',0.01,10.0,1.0,0.05)
    conf = st.sidebar.slider('置信度筛选:',0.01,1.0,0.1,0.05)
    top = st.sidebar.slider('显示数量:',1,20,5,1)
    st.subheader("推荐模型结果:")
    #避免现实'frozenset'
    rules = rules.applymap(lambda x: tuple(x) if isinstance(x, frozenset) else x )
    # lift 提升度首先要大于1，然后再排序选择自己希望深究的前 n 个
    hubu = rules[(rules['lift'] > lift) & (rules['confidence']>=conf)].sort_values('lift', ascending=False)
    st.write('互补品共计:', hubu.shape[0])
    st.table(hubu.head(top))
    huchi = rules[(rules['lift'] <= lift)].sort_values(by='lift', ascending=True)
    st.write('互斥品共计:', huchi.shape[0])
    st.table(huchi.head(top))
    return rules[(rules['lift'] > lift) & (rules['confidence']>=conf)].sort_values('lift', ascending=False)

import base64
from io import BytesIO

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

##########
# main() #
##########
st.sidebar.header('智能推荐系统')
data_load_state = st.text('加载数据...')
df = load_data(DataFile)
data_load_state.text('加载数据完成!')
data_status(df)
result = suggestion_model(df)
st.markdown(get_table_download_link(result), unsafe_allow_html=True)

