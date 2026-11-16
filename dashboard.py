import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

st.set_page_config(page_title="E-Commerce Dashboard", page_icon="📦", layout="wide")
sns.set_theme(style="whitegrid")

MAIN = "#2E6DB4"
GREY = "#C9CED6"
ACCENT = "#D1495B"
DATA_PATH = os.path.join(os.path.dirname(__file__), "main_data.csv")


# ---------- Helper functions ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        df[col] = pd.to_datetime(df[col])
    return df


def create_monthly_df(df):
    monthly = df.resample("ME", on="order_purchase_timestamp").agg(
        order_count=("order_id", "nunique"), revenue=("price", "sum")
    ).reset_index()
    monthly["month"] = monthly["order_purchase_timestamp"].dt.strftime("%Y-%m")
    return monthly


def create_category_df(df):
    cat = df.groupby("category").agg(revenue=("price", "sum")).sort_values("revenue", ascending=False).reset_index()
    cat["share_pct"] = cat["revenue"] / cat["revenue"].sum() * 100
    return cat


def create_order_level_df(df):
    return df.drop_duplicates("order_id")


def create_state_df(order_df):
    return order_df.groupby("customer_state").agg(
        order_count=("order_id", "nunique"),
        avg_delivery_days=("delivery_days", "mean"),
        late_rate=("is_late", "mean"),
        avg_review=("review_score", "mean"),
    ).reset_index()


def create_rfm_df(df):
    reference_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("price", "sum"),
    ).reset_index()
    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days
    rfm["R_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["F_score"] = np.select([rfm["frequency"] >= 3, rfm["frequency"] == 2], [5, 3], default=1)

    conditions = [
        (rfm["F_score"] >= 3) & (rfm["R_score"] >= 4),
        rfm["F_score"] >= 3,
        (rfm["R_score"] >= 4) & (rfm["M_score"] >= 4),
        rfm["R_score"] >= 4,
        (rfm["R_score"] <= 2) & (rfm["M_score"] >= 4),
        rfm["R_score"] <= 2,
    ]
    choices = ["Champions", "Loyal Customers", "Promising", "New Customers", "At Risk", "Hibernating"]
    rfm["segment"] = np.select(conditions, choices, default="Need Attention")
    return rfm


def create_delivery_group_df(order_df):
    bins = [0, 7, 14, 21, 30, np.inf]
    labels = ["<= 7 hari", "8-14 hari", "15-21 hari", "22-30 hari", "> 30 hari"]
    grouped = order_df.assign(delivery_group=pd.cut(order_df["delivery_days"], bins=bins, labels=labels))
    out = grouped.groupby("delivery_group", observed=True).agg(
        order_count=("order_id", "nunique"), avg_review=("review_score", "mean")
    ).reset_index()
    out["delivery_group"] = out["delivery_group"].astype(str)
    return out


def brl(value):
    return format_currency(value, "BRL", locale="pt_BR")


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---------- Load & filter ----------
all_df = load_data()
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.header("📦 E-Commerce Brasil")
    st.caption("Data pesanan delivered, Jan 2017 sampai Agu 2018")
    date_range = st.date_input("Rentang waktu pembelian", value=[min_date, max_date],
                               min_value=min_date, max_value=max_date)
    states = sorted(all_df["customer_state"].dropna().unique())
    selected_states = st.multiselect("Negara bagian pelanggan", states, placeholder="Semua negara bagian")

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

main_df = all_df[(all_df["order_purchase_timestamp"].dt.date >= start_date) &
                 (all_df["order_purchase_timestamp"].dt.date <= end_date)]
if selected_states:
    main_df = main_df[main_df["customer_state"].isin(selected_states)]

if main_df.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Ubah rentang waktu atau negara bagian.")
    st.stop()

order_df = create_order_level_df(main_df)
monthly_df = create_monthly_df(main_df)
category_df = create_category_df(main_df)
state_df = create_state_df(order_df)
rfm_df = create_rfm_df(main_df)
delivery_group_df = create_delivery_group_df(order_df)

# ---------- Header & KPI ----------
st.title("E-Commerce Public Dataset Dashboard")
st.caption(f"Periode {start_date:%d %b %Y} sampai {end_date:%d %b %Y}"
           + (f" | Negara bagian: {', '.join(selected_states)}" if selected_states else ""))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total pesanan", f"{order_df['order_id'].nunique():,}")
total_rev = main_df["price"].sum()
c2.metric("Total revenue", f"R$ {total_rev/1e6:,.2f} jt" if total_rev >= 1e6 else brl(total_rev),
          help=f"Total revenue: {brl(total_rev)}")
c3.metric("Rata-rata review", f"{order_df['review_score'].mean():.2f} / 5")
c4.metric("Pesanan terlambat", f"{order_df['is_late'].mean():.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["Tren Penjualan", "Kategori Produk", "Pengiriman & Review", "Pelanggan (RFM & Peta)"])

# ---------- Tab 1: monthly trend ----------
with tab1:
    st.subheader("Tren jumlah pesanan dan revenue bulanan")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(monthly_df["month"], monthly_df["order_count"], marker="o", linewidth=2, color=MAIN)
    axes[0].set_ylabel("Jumlah pesanan")
    axes[0].set_ylim(0, monthly_df["order_count"].max() * 1.15)
    axes[0].set_title("Jumlah pesanan per bulan", loc="left")

    peak = monthly_df["revenue"].idxmax()
    axes[1].plot(monthly_df["month"], monthly_df["revenue"] / 1e3, marker="o", linewidth=2, color=MAIN)
    axes[1].scatter(monthly_df.loc[peak, "month"], monthly_df.loc[peak, "revenue"] / 1e3, s=110, color=ACCENT, zorder=3)
    axes[1].set_ylabel("Revenue (ribu BRL)")
    axes[1].set_ylim(0, monthly_df["revenue"].max() / 1e3 * 1.15)
    axes[1].set_title("Revenue per bulan (ribu BRL), titik merah = bulan tertinggi", loc="left")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        tidy(ax)
    plt.tight_layout()
    st.pyplot(fig)
    st.info(f"Revenue tertinggi pada **{monthly_df.loc[peak, 'month']}** sebesar "
            f"**{brl(monthly_df.loc[peak, 'revenue'])}**.")

# ---------- Tab 2: categories ----------
with tab2:
    st.subheader("Kategori produk dengan revenue tertinggi dan terendah")
    n_top = st.slider("Jumlah kategori teratas", 5, 20, 10)
    top = category_df.head(n_top)
    bottom = category_df[category_df["category"] != "unknown"].tail(5).sort_values("revenue")

    col1, col2 = st.columns([1.4, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 0.45 * n_top + 1.5))
        colors = [MAIN] + [GREY] * (len(top) - 1)
        sns.barplot(data=top, x="revenue", y="category", hue="category", palette=colors, legend=False, ax=ax)
        for i, (rev, share) in enumerate(zip(top["revenue"], top["share_pct"])):
            ax.text(rev + top["revenue"].max() * 0.01, i, f"{share:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, top["revenue"].max() * 1.15)
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e3:,.0f} rb"))
        ax.set_xlabel("Revenue (BRL)")
        ax.set_ylabel(None)
        ax.set_title(f"{n_top} kategori teratas (label = % dari total revenue)", loc="left")
        tidy(ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=bottom, x="revenue", y="category", color=GREY, ax=ax)
        for i, rev in enumerate(bottom["revenue"]):
            ax.text(rev * 1.02, i, f"BRL {rev:,.0f}", va="center", fontsize=9)
        ax.set_xlim(0, bottom["revenue"].max() * 1.4)
        ax.set_xlabel("Revenue (BRL)")
        ax.set_ylabel(None)
        ax.set_title("5 kategori terendah", loc="left")
        tidy(ax)
        st.pyplot(fig)
    st.info(f"{n_top} kategori teratas menyumbang **{top['share_pct'].sum():.1f}%** dari total revenue.")

# ---------- Tab 3: delivery & review ----------
with tab3:
    st.subheader("Pengaruh keterlambatan pengiriman terhadap review")
    late_df = order_df.groupby("is_late").agg(avg_review=("review_score", "mean")).reset_index()
    late_df["status"] = late_df["is_late"].map({False: "Tepat waktu", True: "Terlambat"})

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=late_df, x="status", y="avg_review", hue="status",
                    palette={"Tepat waktu": MAIN, "Terlambat": ACCENT}, legend=False, ax=ax)
        for i, v in enumerate(late_df["avg_review"]):
            ax.text(i, v + 0.08, f"{v:.2f}", ha="center")
        ax.set_ylim(0, 5)
        ax.set_xlabel(None)
        ax.set_ylabel("Review score (1-5)")
        ax.set_title("Rata-rata review: tepat waktu dan terlambat", loc="left")
        tidy(ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = [MAIN, MAIN, MAIN, GREY, ACCENT][:len(delivery_group_df)]
        sns.barplot(data=delivery_group_df, x="delivery_group", y="avg_review", hue="delivery_group",
                    palette=colors, legend=False, ax=ax)
        for i, v in enumerate(delivery_group_df["avg_review"]):
            ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_ylim(0, 5)
        ax.set_xlabel("Lama pengiriman")
        ax.set_ylabel("Review score (1-5)")
        ax.set_title("Review per kelompok waktu pengiriman (binning)", loc="left")
        tidy(ax)
        st.pyplot(fig)

    st.subheader("Rata-rata waktu pengiriman per negara bagian")
    state_sorted = state_df.sort_values("avg_delivery_days", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = [MAIN if i < 5 else GREY for i in range(len(state_sorted))]
    sns.barplot(data=state_sorted, x="customer_state", y="avg_delivery_days", hue="customer_state",
                palette=colors, legend=False, ax=ax)
    ax.axhline(order_df["delivery_days"].mean(), color="#555", linestyle="--", linewidth=1,
               label=f"Rata-rata: {order_df['delivery_days'].mean():.1f} hari")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel("Negara bagian")
    ax.set_ylabel("Hari (pembelian sampai diterima)")
    tidy(ax)
    st.pyplot(fig)
    with st.expander("Lihat tabel per negara bagian"):
        st.dataframe(state_sorted.rename(columns={
            "customer_state": "Negara bagian", "order_count": "Pesanan", "avg_delivery_days": "Rata-rata hari",
            "late_rate": "Tingkat terlambat", "avg_review": "Rata-rata review"}).round(2), width="stretch")

# ---------- Tab 4: RFM & map ----------
with tab4:
    st.subheader("Segmentasi pelanggan dengan RFM")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rata-rata recency", f"{rfm_df['recency'].mean():.0f} hari")
    c2.metric("Rata-rata frequency", f"{rfm_df['frequency'].mean():.2f} pesanan")
    c3.metric("Rata-rata monetary", brl(rfm_df["monetary"].mean()))

    seg = rfm_df.groupby("segment").agg(customers=("customer_unique_id", "nunique")).sort_values("customers", ascending=False).reset_index()
    seg["share"] = seg["customers"] / seg["customers"].sum() * 100
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = [ACCENT if s == "At Risk" else (MAIN if s in ["Champions", "Loyal Customers"] else GREY) for s in seg["segment"]]
    sns.barplot(data=seg, x="customers", y="segment", hue="segment", palette=colors, legend=False, ax=ax)
    for i, (n, pct) in enumerate(zip(seg["customers"], seg["share"])):
        ax.text(n + seg["customers"].max() * 0.01, i, f"{n:,} ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, seg["customers"].max() * 1.2)
    ax.set_xlabel("Jumlah pelanggan unik")
    ax.set_ylabel(None)
    ax.set_title("Biru = pelanggan berulang, merah = bernilai tinggi tapi lama tidak belanja", loc="left")
    tidy(ax)
    st.pyplot(fig)
    st.caption("Skor R dan M dibuat dengan binning kuantil (1-5); skor F manual: 1 pesanan = 1, 2 pesanan = 3, 3 pesanan atau lebih = 5.")

    st.subheader("Sebaran lokasi pelanggan")
    map_df = (main_df.dropna(subset=["lat", "lng"]).drop_duplicates("customer_unique_id")[["lat", "lng"]]
              .rename(columns={"lng": "lon"}))
    if len(map_df) > 20000:
        map_df = map_df.sample(20000, random_state=42)
    st.map(map_df, size=500, color="#2E6DB466")
    st.caption("Setiap titik mewakili lokasi kode pos pelanggan (maks. 20.000 titik sampel).")

st.caption("Sumber data: Brazilian E-Commerce Public Dataset by Olist | Dibuat oleh Jen Agresia Misti")
