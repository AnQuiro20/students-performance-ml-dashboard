import plotly.express as px


def render_kpis(df, st):
    col1, col2, col3 = st.columns(3)
    col1.metric("Math Average", f"{df['math score'].mean():.2f}")
    col2.metric("Reading Average", f"{df['reading score'].mean():.2f}")
    col3.metric("Writing Average", f"{df['writing score'].mean():.2f}")


def render_gender_gap(df, st):
    gender_avg = df.groupby("gender")["math score"].mean().reset_index()

    fig = px.bar(
        gender_avg,
        x="gender",
        y="math score",
        color="gender",
        template="plotly_white",
        title="Average Math Score by Gender"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_distribution(df, st):
    fig = px.histogram(
        df,
        x="math score",
        nbins=20,
        marginal="box",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_correlation(df, st):
    corr_matrix = df[
        ["math score", "reading score", "writing score"]
    ].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
