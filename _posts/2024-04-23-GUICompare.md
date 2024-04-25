## Second Post
This is my second update.
Generated minutiae and local structures can now be saved into database folders as .npz files.
This also facilitates database comparison, whose output is dispalyed on the GUI via a label.

```       
    def enroll(self):
        self.enroll_b.configure(state=DISABLED)
        if (self.name.get() != ''):
            path = "./samples/" + self.name.get()
        else:
            path = "./samples/" + Path(self.path).stem
        self.image.save(f"{path}.png")
        np.savez(path, self.valid_minutiae, self.local_structures)
        self.name.set('')       
```
```
def local_struct_compare(self):
        f1, m1, ls1 = self.fingerprint, self.valid_minutiae, self.local_structures

        self.positives = []
        self.pos_len=0
        #print("Threshold:", self.threshold)

        filelist=os.listdir('database')
        for file in filelist:
            if file.endswith('.png'):
                ofn = pathlib.Path(file)
                ofn = ofn.with_suffix('')
                
                f2, (m2, ls2) = cv.imread(f'./database/{ofn}.png', cv.IMREAD_GRAYSCALE), np.load(f'./database/{ofn}.npz', allow_pickle=True).values()

                dists = np.linalg.norm(ls1[:,np.newaxis,:] - ls2, axis = -1)
                dists /= np.linalg.norm(ls1, axis = 1)[:,np.newaxis] + np.linalg.norm(ls2, axis = 1)

                num_p = 5 # For simplicity: a fixed number of pairs
                pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
                score = 1 - np.mean(dists[pairs[0], pairs[1]]) # See eq. (23) in MCC paper
                #print(f'Comparison score: {score:.2f}' + ' for ', ofn)

                if (score > self.threshold):
                    self.positives.append([ofn, round(score,2)])
                    #print("Found match: ", ofn)
                    
        
        #print(len(positives))
        text = "Threshold: " + str(self.threshold) + '\n'
        for index in range(len(self.positives)):
            text += "Found match: "
            text += str(self.positives[index][0])
            text += " at score: "
            text += str(self.positives[index][1])
            text += "\n"
            pass
            #print(positives[index])
        self.output_l.configure(text=text)

        self.pos_len=len(self.positives)
```
