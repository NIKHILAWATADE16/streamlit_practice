from email.policy import default
from multiprocessing.sharedctypes import Value
import this
import streamlit as st
import pandas as pd
import numpy as np
import time as time
import matplotlib.pyplot as plt
import matplotlib
import altair as alt
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, peak_widths
st.set_page_config(
    page_title="Dashboard",
    page_icon="✅",
    layout="wide",
)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("HELlO")
st.subheader("HELLOOOOO")
st.markdown("""
# h1
## h2
### h3
#### h4
###### h5
:moon:
--""",True)
st.latex(r"a1+a2=a4")
# st.write(st)
thisdict = {
  "brand": "Ford",
  "electric": False,
  "year": 1964,
  "colors": ["red", "white", "blue"]
}
s={"eloo im Nikhil Awatade from pict":"Pune"}
st.json(thisdict)
st.json(s)

@st.cache
def rt():
    time.sleep(5)
    return time.time()

if st.checkbox("1"):
    st.write(rt())
if st.checkbox("2"):
    st.write(rt())

data=pd.DataFrame(np.random.rand(100,3),columns=['a','b','c'])

st.graphviz_chart("""
digraph{
    Nikhil -> Abhishek
    Ameya -> Abhishek
    Anand -> Abhishek
}""")


# plt.scatter(data['a'],data['b'])
# st.set_option('deprecation.showPyplotGlobalUse', False)
# plt.title("FIRST SCATTER PLOT")
# st.pyplot()

# chart=alt.Chart(data).mark_circle().encode(
#     x='a',y='b',tooltip=['a','b']
#     )
# st.altair_chart(chart)

# st.line_chart(data)
# st.bar_chart(data)
# st.area_chart(data)

# if st.checkbox("Inside button"):
#   if st.button("PLOT"):
#     st.bar_chart(data)


# var=st.radio("City",["Pune","Solapur","Kolhapur","Nagar"])
# var1=st.selectbox("City",[" ","Pune","Solapur","Kolhapur","Nagar"])
# var2=st.multiselect("City",[" ","Pune","Solapur","Kolhapur","Nagar"])

# st.slider("age",min_value=10,max_value=80)
# st.number_input("NUMBERS",min_value=2.0,max_value=99.9,step=1.2)

# var4=st.file_uploader("Upload Image")
# if st.button("Show") and var4:
#   st.image(var4)

# plt.style.use("ggplot")

# rad=st.sidebar.radio("Navigation",["Home","Contact","About us"])
# if rad=="Home":
#   dta={
#     "Linear":[x for x in range(1,50)],
#     "Quad":[x**2 for x in range(1,50)],
#     "Expo":[x**x for x in range(1,50)],
#     "Thrice":[x*3 for x in range(1,50)]
#   }
#   dfa=pd.DataFrame(data=dta)

#   col = st.sidebar.multiselect("Select column",dfa.columns)
#   plt.plot(dfa["Linear"],dfa[col])
#   st.pyplot()

# else:
#   st.error("ERROR OCCURED")
#   st.success("WELL DONE!")
#   st.exception("DIVIDE BY ZERO ERROR")
#   st.info("HAPPY TO INFORM YOU")
#   st.write("Helo")
#   progress=st.progress(0)
#   for i in range (100):
#     time.sleep(0.001)
#     progress.progress(i+1)
#   st.balloons()



# progress_bar = st.progress(0)
# status_text = st.empty()
# chart = st.line_chart(np.random.randn(10, 2))

# for i in range(100):
#     # Update progress bar.
#     progress_bar.progress(i + 1)

#     new_rows = np.random.randn(10, 2)

#     # Update status text.
#     status_text.text('The latest random number is: %s' % new_rows[-1, 1])

#     # Append data to the chart.
#     chart.add_rows(new_rows)

#     # Pretend we're doing some computation that takes time.
#     time.sleep(1)

# status_text.text('Done!')
# st.balloons()


# f,s=st.columns(2)
# f=st.bar_chart(pd.DataFrame(np.random.rand(100,2)))
# # s=st.button("Push")

# y=np.random.normal(5,15,size=7)
# fig,x=plt.subplots()
# x.hist(y,bins=10)
# fig


# dataset_url = "https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv"

# # read csv from a URL
# @st.experimental_memo
# def get_data() -> pd.DataFrame:
#     return pd.read_csv(dataset_url)

# df = get_data()
# st.write("nikhil")


# eda = pd.read_csv("C:\\Users\\NIKHIL\\OneDrive\\Desktop\\fricorp\\eda.csv")
# st.write(eda.describe())



