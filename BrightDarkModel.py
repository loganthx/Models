from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np, cv2 as cv, matplotlib.pyplot as plt, random, time

def label(img):
	M = np.mean([pixel for pixel in img])
	if M >= .52: label = 0 #'Bright'
	else: label = 1 #'Dark'
	return label
def binary2Label(val): 
	if val==1: lbl = 'Dark'
	else: lbl = 'Bright'
	return lbl

spec = np.arange(.1, .9, .0001)
imgs = [np.minimum( 1, abs(np.random.normal(i, .2, (50,50,1) ))) for i in spec]
imgs_labels = [([i, label(i)]) for i in imgs]
B_data, D_data = [i for i in imgs_labels if i[1]==0], [i for i in imgs_labels if i[1] == 1]
Cut = min([len(B_data), len(D_data)])

dataset = B_data[:Cut] + D_data[:Cut]
random.shuffle(dataset)
X, y = np.array([i[0] for i in dataset]).reshape(-1, 50, 50, 1), np.array([i[1] for i in dataset])

model = Sequential()
model.add(Flatten(input_shape=(50,50)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=['accuracy'])
model.fit(X, y, batch_size=8, epochs=4, validation_split=0.2)

rn = np.arange(.1, .9, .01)
Val = np.array([np.minimum( 1, abs(np.random.normal(i, .2, (50,50,1) ))) for i in rn])
preds = [binary2Label(np.around(model.predict([i.reshape(-1, 50, 50, 1)]))) for i in Val]
results = [[img, pred] for img, pred in zip(Val, preds)]
random.shuffle(results)

axs = plt.subplots(4, 6, figsize=(8, 8))[-1].flatten()
time.sleep(1)
for (img, pred), ax in zip(results[:30], axs):
	ax.imshow(img, cmap='gray')
	ax.set_title(f"{pred}")
	ax.axis('off')	
plt.show()