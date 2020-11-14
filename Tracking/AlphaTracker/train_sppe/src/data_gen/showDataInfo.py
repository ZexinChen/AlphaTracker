import h5py
annot = h5py.File('data.h5')
for k in annot.keys():
	print(k)

bndboxes = annot['bndbox'][:]
print(bndboxes.shape)
mgnames = annot['imgname'][:]
print(mgnames.shape)
parts = annot['part'][:]
print(parts.shape)
