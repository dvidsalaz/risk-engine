import streamlit as st
import pandas as pd

import numpy as np

st.write("""
# Home
Hello *world!*
""")

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})



st.number_input("""#test""")



print(np.__version__)

a = np.array([[1, 2, 3],
              [4, 5, 6]])
a.shape
(2, 3)