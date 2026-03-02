import matplotlib.pyplot as plt
def plot_distr_unif_exp_norm(X,distribucion, n_bins = 50, save_fig = False):
    # distribucion {'norm', 'exp','unif'}
    if distribucion == 'unif':
        n_axes= ['PRI_jet_leading_phi', 'PRI_jet_subleading_phi']
        fig, axes = plt.subplots(nrows=1, ncols=2,  figsize=(10, 4), tight_layout=True)
        plt.rcParams.update({'font.size': 12})
    if distribucion == 'norm':
        n_axes=['PRI_jet_leading_eta','PRI_jet_subleading_eta']
        fig, axes = plt.subplots(nrows=1, ncols=2,  figsize=(10, 4), tight_layout=True)
        plt.rcParams.update({'font.size': 12})
    if distribucion == 'exp':
        n_axes= ['PRI_jet_leading_pt','PRI_jet_subleading_pt','DER_mass_jet_jet']
        fig, axes = plt.subplots(nrows=1, ncols=3,  figsize=(10, 4), tight_layout=True)
        plt.rcParams.update({'font.size': 10})
    a=0
    for ax in n_axes: 
        axes1 = X[ax]

        axes[a].hist(axes1, bins=n_bins)
        axes[a].set_title("Distribución de" + ax)
        axes[a].set_xlabel('Valor')
        axes[a].set_ylabel('Frecuencia')
        a+=1
    if (save_fig == True) and (distribucion == 'unif'):
        plt.savefig('Fotos-A-Datos/Distribucion_uniforme.png')
    if (save_fig == True) and (distribucion == 'norm'):
        plt.savefig('Fotos-A-Datos/Distribucion_normal.png')
    if (save_fig == True) and (distribucion == 'exp'):
        plt.savefig('Fotos-A-Datos/Distribucion_exponencial.png')
        
def imprimir_graficas_de_distribuciones(X, save_fig = False):
    attrs = ['DER_lep_eta_centrality', 'DER_prodeta_jet_jet', 'DER_deltaeta_jet_jet', 'DER_mass_MMC']
    for attr in attrs:
        if attr != 'DER_mass_MMC':
            attr_i = X[attr]
            attr_i.hist(bins=100)
            plt.title('Distribución de '+ attr )
            plt.xlabel('Valor')
            plt.ylabel('Frecuencia')
            if save_fig == True:
                plt.savefig('Fotos-A-Datos/'+ attr +'.png')
            plt.show()

        else:
            plt.figure(figsize=(16,10))
            X['DER_mass_MMC'].hist(bins=500)
            plt.title('Distribución de DER_mass_MMC')
            plt.xlabel('Valor')
            plt.ylabel('Frecuencia')

            media = X['DER_mass_MMC'].mean()
            mediana = X['DER_mass_MMC'].median()

            plt.axvline(media, color='r', linestyle='dashed', linewidth=2, label=f'Media: {media:.2f}')
            plt.axvline(mediana, color='g', linestyle='dashed', linewidth=2, label=f'Mediana: {mediana:.2f}')

            if save_fig == True:
                plt.savefig('Fotos-A-Datos/'+ attr +'.png')
            plt.legend()
            plt.show()