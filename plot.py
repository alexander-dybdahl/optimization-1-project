import matplotlib.pyplot as plt

# plotting function
def plot_3D(X, edges, p, title, angles = [(0, 30), (20, 30), (90, 30)]):
    edges_cab, edges_bar = edges
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, fontsize=20)
    M = len(p)
    for i, angle in enumerate(angles):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.view_init(*angle)
        ax.scatter(X[M:,0], X[M:,1], X[M:,2], c='r', s=5, label='free nodes')
        for j, txt in enumerate(range(M,len(X))):
            ax.text(X[j,0], X[j,1], X[j,2], txt+1, fontsize=10)
        if M != 0:
            ax.scatter(p[:,0], p[:,1], p[:,2], c='b', s=5, label='fixed nodes')
        a, b = True, True
        for i, j in edges_cab:
            if a:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], [X[i,2], X[j,2]], '--', c='b', linewidth=0.5, label='cable')
                a=False
            else:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], [X[i,2], X[j,2]], '--', c='b', linewidth=0.5)
        for i, j in edges_bar:
            if b:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], [X[i,2], X[j,2]], '-', c='g', linewidth=0.9, label='bar')
                b=False
            else:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], [X[i,2], X[j,2]], '-', c='g', linewidth=0.9)
    plt.tight_layout()
    plt.legend()
    plt.show()

# plotting function
def plot_system(X, X_init, X_analytic, conv, edges, p):
    edges_cab, edges_bar = edges
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle("Tensegrity Structure", fontsize=20)
    M = len(p)
    Xs = (X_init, X, X_analytic)
    labs = ('Initial', 'Optimized', 'Analytical')
    for n, X in enumerate(Xs):
        ax = fig.add_subplot(1, 4, n+1, projection='3d')
        ax.scatter(X[M:,0], X[M:,1], X[M:,2], c='r', s=5, label='free nodes')
        ax.set_title(labs[n])
        for j, txt in enumerate(range(M,len(X))):
            ax.text(X[j+M,0], X[j+M,1], X[j+M,2], txt+1, fontsize=10)
        if M != 0:
            ax.scatter(p[:,0], p[:,1], p[:,2], c='b', s=5, label='fixed nodes')
            for j, txt in enumerate(range(M)):
                ax.text(p[j,0], p[j,1], p[j,2], txt+1, fontsize=10)
        a, b = True, True
        for i, j in edges_cab:
            if a:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], [X[i,2], X[j,2]], '--', c='b', linewidth=0.5, label='cable')
                a=False
            else:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], [X[i,2], X[j,2]], '--', c='b', linewidth=0.5)
        for i, j in edges_bar:
            if b:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], [X[i,2], X[j,2]], '-', c='g', linewidth=0.9, label='bar')
                b=False
            else:
                ax.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], [X[i,2], X[j,2]], '-', c='g', linewidth=0.9)    
        plt.legend(loc='upper right')
    ax = fig.add_subplot(1, 4, 4)
    ax.loglog(conv, label='$\\nabla E(X)$')
    ax.set_title('Convergence')

    plt.tight_layout()
    plt.show()
