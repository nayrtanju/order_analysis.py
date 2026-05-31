import streamlit as st
import tempfile
import os
import traceback
import matplotlib.pyplot as plt
import numpy as np

st.title("Order Map Analysis")
st.write("App started successfully")

try:
    from order_analysis import read_xlsx_numeric, angular_resample, order_map, extract_order_vs_rpm
    st.success("order_analysis.py başarıyla yüklendi")
except Exception as e:
    st.error("order_analysis.py yüklenirken hata oluştu")
    st.code(traceback.format_exc())
    st.stop()

uploaded_file = st.file_uploader("Excel dosyası yükle", type=["xlsx"])

if uploaded_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded_file.read())
            xlsx_path = tmp.name

        headers, data = read_xlsx_numeric(xlsx_path)

        st.write("Excel başarıyla okundu")
        st.write("Headers:", headers)
        st.write("Data shape:", data.shape)

        time = data[:, 0]
        rpm = data[:, 4]

        channels = {
            "ChA": data[:, 1],
            "ChB": data[:, 2],
            "ChC": data[:, 3],
        }

        selected_channel = st.selectbox("Order Map kanalı", list(channels.keys()))

        samples_per_rev = st.slider("Samples per revolution", 128, 2048, 512)
        max_order = st.slider("Max order", 5, 50, 30)
        target_order = st.number_input("Çizilecek order", value=10.0)

        if st.button("Order Map Oluştur"):
            with st.spinner("Order Map hesaplanıyor..."):
                sig = channels[selected_channel]

                theta_u, x_u, rpm_u = angular_resample(
                    time, rpm, sig, samples_per_rev=samples_per_rev
                )

                orders, rpms, spec = order_map(
                    theta_u,
                    x_u,
                    rpm_u,
                    samples_per_rev=samples_per_rev,
                    max_order=max_order
                )

                idx = np.argsort(rpms)
                r = rpms[idx]
                s = spec[idx]

                db = 20 * np.log10(np.maximum(s, 1e-12))

                fig, ax = plt.subplots(figsize=(11, 7))
                im = ax.imshow(
                    db,
                    aspect="auto",
                    origin="lower",
                    extent=[orders[0], orders[-1], r[0], r[-1]],
                    interpolation="nearest"
                )

                fig.colorbar(im, ax=ax, label="Amplitude [dB re 1 m/s²]")
                ax.set_xlabel("Order")
                ax.set_ylabel("RPM")
                ax.set_title(f"Order Map - {selected_channel}")

                st.pyplot(fig)

                st.subheader(f"{target_order}. Order vs RPM - Tüm Kanallar")

                fig2, ax2 = plt.subplots(figsize=(11, 7))

                for name, sig in channels.items():
                    theta_u, x_u, rpm_u = angular_resample(
                        time, rpm, sig, samples_per_rev=samples_per_rev
                    )

                    orders, rpms, spec = order_map(
                        theta_u,
                        x_u,
                        rpm_u,
                        samples_per_rev=samples_per_rev,
                        max_order=max_order
                    )
rpm_sorted, amp_sorted = extract_order_vs_rpm(
    orders,
    rpms,
    spec,
    target_order=target_order,
    smooth=True
)

ax2.plot(
    rpm_sorted,
    amp_sorted,
    label=name
)

                ax2.set_xlabel("RPM")
                ax2.set_ylabel(f"{target_order}. Order Amplitude [m/s²]")
                ax2.set_title(f"{target_order}. Order vs RPM - All Channels")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                st.pyplot(fig2)

        os.remove(xlsx_path)

    except Exception:
        st.error("Uygulama çalışırken hata oluştu")
        st.code(traceback.format_exc())
