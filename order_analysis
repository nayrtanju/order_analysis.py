import zipfile, xml.etree.ElementTree as ET, re, gc, os
import numpy as np
import matplotlib.pyplot as plt

XLSX = '/mnt/data/ConvertedData.xlsx'
OUTDIR = '/mnt/data/order_outputs'
os.makedirs(OUTDIR, exist_ok=True)
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

def load_shared_strings(z):
    strings=[]
    if 'xl/sharedStrings.xml' not in z.namelist():
        return strings
    for event, elem in ET.iterparse(z.open('xl/sharedStrings.xml'), events=('end',)):
        if elem.tag == NS+'si':
            texts=[]
            for t in elem.iter(NS+'t'):
                texts.append(t.text or '')
            strings.append(''.join(texts))
            elem.clear()
    return strings

def col_index(cell_ref):
    letters = re.match(r'([A-Z]+)', cell_ref).group(1)
    n=0
    for ch in letters:
        n=n*26+ord(ch)-64
    return n-1

def read_xlsx_numeric(path):
    with zipfile.ZipFile(path) as z:
        shared = load_shared_strings(z)
        # known dimension
        nrows=0
        with z.open('xl/worksheets/sheet1.xml') as f:
            for ev, elem in ET.iterparse(f, events=('start',)):
                if elem.tag == NS+'dimension':
                    ref=elem.attrib.get('ref','')
                    m=re.search(r':([A-Z]+)(\d+)', ref)
                    nrows=int(m.group(2))-1 if m else 0
                    break
        arr=np.empty((nrows,5), dtype=np.float64)
        headers=[None]*5
        ri=-1
        with z.open('xl/worksheets/sheet1.xml') as f:
            for ev, row in ET.iterparse(f, events=('end',)):
                if row.tag != NS+'row':
                    continue
                rnum=int(row.attrib.get('r','0'))
                vals=[np.nan]*5
                for c in row.findall(NS+'c'):
                    ref=c.attrib.get('r','')
                    j=col_index(ref)
                    if j>=5: continue
                    typ=c.attrib.get('t')
                    v=c.find(NS+'v')
                    if v is None:
                        continue
                    txt=v.text
                    if rnum==1:
                        headers[j]=shared[int(txt)] if typ=='s' else txt
                    else:
                        vals[j]=float(txt)
                if rnum>1:
                    ri+=1
                    arr[ri,:]=vals
                row.clear()
        return headers, arr[:ri+1]

def angular_resample(time, rpm, signal, samples_per_rev=512):
    # Sort/clean
    mask=np.isfinite(time)&np.isfinite(rpm)&np.isfinite(signal)&(rpm>0)
    time=time[mask]; rpm=rpm[mask]; signal=signal[mask]
    dt=np.diff(time, prepend=time[0])
    dt[0]=np.median(np.diff(time[:min(len(time),10000)]))
    omega=2*np.pi*rpm/60.0
    theta=np.cumsum(omega*dt)
    # remove possible non-increasing theta duplicates
    keep=np.r_[True, np.diff(theta)>0]
    theta=theta[keep]; signal=signal[keep]; rpm=rpm[keep]
    dtheta=2*np.pi/samples_per_rev
    theta_u=np.arange(theta[0], theta[-1], dtheta)
    x_u=np.interp(theta_u, theta, signal)
    rpm_u=np.interp(theta_u, theta, rpm)
    return theta_u, x_u, rpm_u

def order_map(theta_u, x_u, rpm_u, samples_per_rev=512, revs_per_block=8, overlap=0.75, max_order=30):
    nper=int(samples_per_rev*revs_per_block)
    hop=max(1,int(nper*(1-overlap)))
    win=np.hanning(nper)
    cg=np.sum(win)
    orders=np.fft.rfftfreq(nper, d=1/samples_per_rev)
    keep=orders<=max_order
    orders=orders[keep]
    specs=[]; rpms=[]
    for start in range(0, len(x_u)-nper+1, hop):
        block=x_u[start:start+nper]
        block=block-np.mean(block)
        amp=2*np.abs(np.fft.rfft(block*win))/cg
        specs.append(amp[keep])
        rpms.append(np.mean(rpm_u[start:start+nper]))
    return orders, np.asarray(rpms), np.asarray(specs)

def plot_order_map(orders, rpms, spec, ch):
    # sort by rpm for nicer y-axis if runup not monotonic
    idx=np.argsort(rpms)
    r=rpms[idx]; s=spec[idx]
    db=20*np.log10(np.maximum(s, 1e-12)/1.0)  # dB re 1 m/s²
    plt.figure(figsize=(11,7))
    extent=[orders[0], orders[-1], r[0], r[-1]]
    plt.imshow(db, aspect='auto', origin='lower', extent=extent, interpolation='nearest')
    plt.colorbar(label='Amplitude [dB re 1 m/s²]')
    plt.xlabel('Order')
    plt.ylabel('RPM')
    plt.title(f'Order Map - {ch}')
    plt.tight_layout()
    path=f'{OUTDIR}/order_map_{ch}.png'
    plt.savefig(path, dpi=180)
    plt.close()
    return path

def main():
    headers, data=read_xlsx_numeric(XLSX)
    print('headers', headers, 'shape', data.shape)
    time=data[:,0]; rpm=data[:,4]
    ch_names=['ChA','ChB','ChC']
    all_results=[]
    map_paths=[]
    for ci,ch in enumerate(ch_names, start=1):
        theta_u,x_u,rpm_u=angular_resample(time, rpm, data[:,ci])
        orders,rpms,spec=order_map(theta_u,x_u,rpm_u)
        map_paths.append(plot_order_map(orders,rpms,spec,ch))
        idx10=int(np.argmin(np.abs(orders-10.0)))
        all_results.append((ch,rpms,spec[:,idx10]))
        # Save channel order map matrix compact npz/csv
        np.savez_compressed(f'{OUTDIR}/order_map_{ch}.npz', orders=orders, rpm=rpms, amplitude=spec)
        del theta_u,x_u,rpm_u,orders,rpms,spec; gc.collect()
    # combine 10th order by nearest common RPM bins
    # concatenate sorted raw data into CSV rows
    import csv
    csv_path=f'{OUTDIR}/order10_all_channels_vs_rpm.csv'
    with open(csv_path,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['Channel','RPM','Order10_Amplitude_m_s2','Order10_dB_re_1_m_s2'])
        for ch,rpms,amp in all_results:
            for rr,aa in zip(rpms,amp):
                w.writerow([ch, rr, aa, 20*np.log10(max(aa,1e-12))])
    plt.figure(figsize=(11,7))
    for ch,rpms,amp in all_results:
        idx=np.argsort(rpms)
        plt.plot(rpms[idx], amp[idx], label=ch, linewidth=1.4)
    plt.xlabel('RPM')
    plt.ylabel('10th order amplitude [m/s²]')
    plt.title('10th Order vs RPM - All Channels')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p10=f'{OUTDIR}/order10_all_channels_vs_rpm.png'
    plt.savefig(p10, dpi=180)
    plt.close()
    # dB plot too
    plt.figure(figsize=(11,7))
    for ch,rpms,amp in all_results:
        idx=np.argsort(rpms)
        plt.plot(rpms[idx], 20*np.log10(np.maximum(amp[idx],1e-12)), label=ch, linewidth=1.4)
    plt.xlabel('RPM')
    plt.ylabel('10th order amplitude [dB re 1 m/s²]')
    plt.title('10th Order vs RPM - All Channels (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p10db=f'{OUTDIR}/order10_all_channels_vs_rpm_dB.png'
    plt.savefig(p10db, dpi=180)
    plt.close()
    print('outputs', map_paths+[p10,p10db,csv_path])
if __name__=='__main__': main()
