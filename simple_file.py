import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
df = pd.read_csv('/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv')
st.title('Amazon Sale Report')
st.header('Progetto per università')
st.write('Lorem Ipsum è un testo segnaposto utilizzato nel settore della tipografia e della stampa. Lorem Ipsum è considerato il testo segnaposto standard sin dal sedicesimo secolo, quando un anonimo tipografo prese una cassetta di caratteri e li assemblò per preparare un testo campione. È sopravvissuto non solo a più di cinque secoli, ma anche al passaggio alla videoimpaginazione, pervenendoci sostanzialmente inalterato. Fu reso popolare, negli anni ’60, con la diffusione dei fogli di caratteri trasferibili “Letraset”, che contenevano passaggi del Lorem Ipsum, e più recentemente da software di impaginazione come Aldus PageMaker, che includeva versioni del Lorem Ipsum.')


col_1, col_2, col_3 = st.columns(3)

with col_1:
    st.write('Lorem Ipsum è un testo segnaposto utilizzato nel settore della tipografia e della stampa. Lorem Ipsum è considerato il testo segnaposto standard sin dal sedicesimo secolo, quando un anonimo tipografo prese una cassetta di caratteri e li assemblò per preparare un testo campione. È sopravvissuto non solo a più di cinque secoli, ma anche al passaggio alla videoimpaginazione, pervenendoci sostanzialmente inalterato. Fu reso popolare, negli anni ’60, con la diffusione dei fogli di caratteri trasferibili “Letraset”, che contenevano passaggi del Lorem Ipsum, e più recentemente da software di impaginazione come Aldus PageMaker, che includeva versioni del Lorem Ipsum.')
with col_2:
    st.write(df)

st.sidebar.header('Settings')
if st.sidebar.checkbox('Show dataframe'):
    st.write(df.head())

st.sidebar.code('''import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st''')

box_select = st.selectbox('select feature',['Qty','Amount'])

if box_select == 'Qty':
    fig, ax = plt.subplots()
    ax.hist(df.Qty)
    st.write(fig)
else:
    fig, ax = plt.subplots()
    ax.hist(df.Amount)
    st.write(fig)