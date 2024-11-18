import torch
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def window(img, ww, wl, rescale_intercept=0):
    """
    Takes single CT image in HU range. Applies windowing. 
    Scales each image to (0,1) without clipping.
    
    Parameters
    ----------
    img : ndarray
    
    Returns
    -------
    img : ndarray
    """
    img = (img - (wl-ww/2))/ww
    return img

class SparseDataset2D(torch.utils.data.Dataset):

    """
    Dataset for the sparse-view artifact removal task.     
        
    Parameters
    ----------
    df : dataframe
        dataframe with the traing/val/test data
    path_sparse : string
        path of the directory where the sparse-view data is located
    path_gt : string
        path of the directory where the full-view data is located
    augmentation : bool
        If data augmentation shall be applied or not.
    image_size : touple
        size, to which the images are cut if specified, default: original size of image
    ww, wl : int
        window width and level to which the input images are clipped
    rescale_intercept : int
        to get to HU values

    Returns
    -------
    img : ndarray
    labels : list
    """
        
    def __init__(self, 
                 df = None, 
                 path_sparse = None, 
                 path_gt = None, 
                 augmentation = False, 
                 image_size=None, 
                 ww=None, 
                 wl=None,
                 rescale_intercept=0
                ):

        self.df = df.copy()
        self.filename_list = self.df['SOPInstanceUID'].values
        self.path_sparse = path_sparse
        self.path_gt = path_gt
        self.augmentation = augmentation
        self.image_size = image_size
        self.ww = ww
        self.wl = wl
        self.rescale_intercept=rescale_intercept

    def __len__(self):
        return int(len(self.filename_list))

    def __getitem__(self, idx):
        
        filename = self.filename_list[idx % len(self.filename_list)]
        image_sparse, image_gt, label = self.__data_gen(filename)
        image_sparse, image_gt = window(image_sparse, ww=self.ww, wl=self.wl, rescale_intercept=self.rescale_intercept), window(image_gt, ww=self.ww, wl=self.wl, rescale_intercept=self.rescale_intercept)
        return image_sparse, image_gt, label

    def load(self, path, filename):
        return np.load(path + "/" + filename + ".npy")

    def _augmentation(self, img1, img2):
        """
        Apply augemntations with certain probabilites specified in the individual functions.
        
        input : array
        out: array
        """
        img = np.concatenate([img1, img2], 0)
        
        #random rotate
        img = self.rot_90(img)

        #random horizontal/vertical flip
        img = self.flip(img)
        
        assert(img.shape==(2, self.image_size[0], self.image_size[1]))
        return img[0:1, :, :].copy(), img[1:2, :, :].copy()
    
    def random_shift(self, img, prob=0.5):
        if random.uniform(0, 1) > prob:
            return img
        
        else:
            shift = (int(random.uniform(-75, 76)), int(random.uniform(-75, 76)))
            img = np.roll(img, shift, axis=(1, 2))
            return img.copy()
        
    def random_erasing(self, img, prob=0.4, lp=0.01, hp=0.2):
        if random.uniform(0, 1) > prob:
            return img
        
        else:
            ereased_area = random.uniform(lp, hp) * img.shape[0]**2
            a = int(random.uniform(1, img.shape[0]-1))
            b = int(ereased_area/a) 

            center = (int(random.uniform(a//2, img.shape[0] - a//2)), int(random.uniform(b//2, img.shape[0] - b//2)))

            img[:, center[0] - a//2:center[0] + a//2, center[1] - b//2:center[1] + b//2] = random.uniform(0, 255)
            return img.copy()

    def rot_90(self, img, prob=0.5):
        if random.uniform(0, 1) > prob:
            return img
        
        else:
            k = random.choice([-1, 1, 2])
            img = np.rot90(img, k=k, axes=(-1, -2)).copy()
            return img.copy()
    
    def flip(self, img, prob=0.5):
        if random.uniform(0, 1) > prob:
            return img
        else:
            horizontal_flip = np.random.choice([1, -1])
            vertical_flip = np.random.choice([1, -1])
            img = img[:, ::horizontal_flip, ::vertical_flip]
            return img.copy()
            
    def noise(self, img, prob=0.9):
        if random.uniform(0, 1) > prob:
            return img
        else:
            sigma_s = 50./255.*np.random.uniform()
            sigma_c = 30./255.*np.random.rand()
        
        
            #poisson noise
            noise_s_map = img*sigma_s**2
            noise_s = np.random.standard_normal((img.shape)) * np.sqrt(noise_s_map)
        
            #gaussian noise
            noise_c_map = np.ones_like(img)*sigma_c
            noise_c = np.random.standard_normal((img.shape)) * noise_c_map
        
            img_noisy = img + noise_s + noise_c
            #print('up')
            '''
            noisy_img = img + \
                np.random.normal(0.0, 1.0, img.shape) * (sigma_s * img) + \
                np.random.normal(0.0, 1.0, img.shape) * sigma_c
            '''
            return img_noisy.copy()
            
    def cut(self, img1, img2):
        """
        randomly cut out a patch from two images at the same location
        """
        xx = np.random.randint(0, img1.shape[1] - self.image_size[0])
        yy = np.random.randint(0, img1.shape[2] - self.image_size[1])
        img1 = img1[:, xx:xx+self.image_size[0], yy:yy+self.image_size[1]]
        img2 = img2[:, xx:xx+self.image_size[0], yy:yy+self.image_size[1]]

        return img1, img2

    def __data_gen(self, filename):
        """
        Generate inpt img
        
        inputs : list
        outputs : array, array
        input and label
        """

        image_sparse, image_gt = self.load(self.path_sparse, filename)[None, :, :], self.load(self.path_gt, filename)[None, :, :]
        if self.image_size is not None:
            image_sparse, image_gt = self.cut(image_sparse, image_gt)

        label = self.df.loc[self.df['SOPInstanceUID']==filename][['pe_present_on_image', 'leftsided_pe', 'central_pe', 'rightsided_pe']].values

        if self.augmentation == True:
            image_sparse, image_gt = self._augmentation(image_sparse, image_gt)
        return image_sparse, image_gt, label.squeeze()


class SparseDataset3D(torch.utils.data.Dataset):

    """
    Dataset for the sparse-view artifact removal task.     
        
    Parameters
    ----------
    df : pandas dataframe
        list with the traing/val/test data
    path_sparse : string
        path of the directory where the sparse-view data is located
    path_gt : string
        path of the directory where the full-view data is located
    augmentation : bool
        If data augmentation shall be applied or not.
    image_size : touple
        size, to which the images are cut if specified, default: original size of image
    ww, wl : int
        window width and level to which the input images are clipped
    rescale_intercept : int
        to get to HU values

    Returns
    -------
    img : ndarray
    labels : list
    """
        
    def __init__(self, 
                 df = None, 
                 path = None, 
                 augmentation = False, 
                 ww=None, 
                 wl=None,
                 rescale_intercept=0,
                 downsample=True,
                 size=(100, 256, 256)):

        self.df = df
        self.filename_list = df["filename"].values
        self.path = path
        self.augmentation = augmentation
        self.ww = ww
        self.wl = wl
        self.rescale_intercept=rescale_intercept
        self.downsample = downsample
        self.size = size

    def __len__(self):
        return int(len(self.filename_list))

    def __getitem__(self, idx):
        filename = self.filename_list[idx % len(self.filename_list)]
        
        fullview, inpt, x1, x2, x3, x4, x5 = self.load(filename)
        return fullview, inpt, x1, x2, x3, x4, x5

    def load(self, filename):
        nr_slices = int(self.df.loc[self.df['filename'] == filename]["nr_slices"].values)
        #randomly chosse one quarter in x-y direction and 100 slices in z direction
        if self.downsample==True:
            rand1, rand2, rand3 = random.choice(range(0, nr_slices-self.size[0])), random.choice([0, 1]), random.choice([0, 1])
        else:
            rand1, rand2, rand3 = random.choice(range(0, nr_slices-self.size[0])), 0, 0
            
        inpt = np.load(self.path + f"/{filename}/{filename}.npy", mmap_mode="c")[None, rand1:rand1+self.size[0], rand2*self.size[1]:(1+rand2)*self.size[1], rand3*self.size[2]:(1+rand3)*self.size[2]]
        inpt = window(inpt, self.ww, self.wl, self.rescale_intercept)
        fullview = np.load(self.path + f"/{filename}/{filename}_fullview.npy", mmap_mode="c")[None, rand1:rand1+self.size[0], rand2*self.size[1]:(1+rand2)*self.size[1], rand3*self.size[2]:(1+rand3)*self.size[2]] 
        fullview = window(fullview, self.ww, self.wl, self.rescale_intercept)
        
        x1 = np.load(self.path + f"/{filename}/{filename}_x1.npy", mmap_mode="c")[:, rand1:rand1+self.size[0], rand2*self.size[1]:(1+rand2)*self.size[1], rand3*self.size[2]:(1+rand3)*self.size[2]] 
        x2 = np.load(self.path + f"/{filename}/{filename}_x2.npy", mmap_mode="c")[:, rand1:rand1+self.size[0], rand2*self.size[1]//2:(1+rand2)*self.size[1]//2, rand3*self.size[2]//2:(1+rand3)*self.size[2]//2]   
        x3 = np.load(self.path + f"/{filename}/{filename}_x3.npy", mmap_mode="c")[:, rand1:rand1+self.size[0], rand2*self.size[1]//4:(1+rand2)*self.size[1]//4, rand3*self.size[2]//4:(1+rand3)*self.size[2]//4] 
        x4 = np.load(self.path + f"/{filename}/{filename}_x4.npy", mmap_mode="c")[:, rand1:rand1+self.size[0], rand2*self.size[1]//8:(1+rand2)*self.size[1]//8, rand3*self.size[2]//8:(1+rand3)*self.size[2]//8] 
        x5 = np.load(self.path + f"/{filename}/{filename}_x5.npy", mmap_mode="c")[:, rand1:rand1+self.size[0], rand2*self.size[1]//16:(1+rand2)*self.size[1]//16, rand3*self.size[2]//16:(1+rand3)*self.size[2]//16] 
        return fullview, inpt, x1, x2, x3, x4, x5

  