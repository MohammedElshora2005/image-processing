import streamlit as st
import io
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ============= إعدادات الصفحة =============
st.set_page_config(
    page_title="Advanced Image Processing Lab",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= دوال مساعدة محسّنة =============
@st.cache_data
def load_image(uploaded_file):
    return np.array(Image.open(uploaded_file))

def display_image(img, caption="Image", use_column_width=True):
    if len(img.shape) == 2:
        st.image(img, caption=caption, use_container_width=use_column_width, channels="GRAY")
    else:
        st.image(img, caption=caption, use_container_width=use_column_width)

def show_comparison(original, processed, title1="Original", title2="Processed"):
    col1, col2 = st.columns(2)
    with col1:
        display_image(original, title1)
    with col2:
        display_image(processed, title2)

# ============= دوال متقدمة =============

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_non_local_means(image, h=10, template_window_size=7, search_window_size=21):
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)

def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners) if corners is not None else []
    result = image.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(result, (x, y), 5, (255, 0, 0), -1)
    return result

def detect_blobs(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    return cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), 
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def apply_canny_edge(image, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def apply_hough_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    result = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return result

def apply_watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    result = image.copy()
    result[markers == -1] = [255, 0, 0]
    return result

def apply_fourier_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
    return magnitude_spectrum

def apply_low_pass_filter_fourier(image, d=30):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - d:crow + d, ccol - d:ccol + d] = 1
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(img_back)

def apply_high_pass_filter_fourier(image, d=30):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - d:crow + d, ccol - d:ccol + d] = 0
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(img_back)

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def clahe_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def pencil_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - gray
    blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

def cartoonize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def invert_colors(image):
    return 255 - image

def sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(image, sepia_filter)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def vignette(image, intensity=0.5):
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/3)
    kernel_y = cv2.getGaussianKernel(rows, rows/3)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = mask ** intensity
    mask = np.dstack([mask] * 3)
    return np.uint8(image * mask)

# ============= الواجهة الرئيسية =============
st.title("🎨 Advanced Image Processing Laboratory")
st.markdown("---")

# شريط جانبي للإعدادات
with st.sidebar:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    
    if uploaded_file is not None:
        img = load_image(uploaded_file)
        st.success(f"✅ Image loaded: {img.shape[1]}x{img.shape[0]}")
        
        # معلومات الصورة
        with st.expander("📊 Image Info"):
            st.write(f"**Dimensions:** {img.shape[0]} x {img.shape[1]}")
            st.write(f"**Channels:** {img.shape[2] if len(img.shape) > 2 else 1}")
            st.write(f"**Data type:** {img.dtype}")
            st.write(f"**Value range:** [{img.min()}, {img.max()}]")

# تبويبات للميزات المختلفة
if uploaded_file is not None:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🎯 Basic Operations", "🌈 Color & Enhancement", "🔍 Filters & Edges", 
        "🧬 Morphology & Segmentation", "📊 Histogram & Threshold", 
        "🎨 Artistic Effects", "⚡ Frequency Domain", "🤖 Advanced Features"
    ])

    # ==================== TAB 1: Basic Operations ====================
    with tab1:
        st.header("Basic Image Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_image(img, "Original Image")
            
            transformation = st.selectbox(
                "Select transformation:",
                ['None', 'Grayscale', 'Solarization', 'Complement', 'Invert', 'Sepia']
            )
            
            if transformation == 'Grayscale':
                result = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            elif transformation == 'Solarization':
                result = np.where(img < 128, 255 - img, img)
            elif transformation == 'Complement':
                result = 255 - img
            elif transformation == 'Invert':
                result = invert_colors(img)
            elif transformation == 'Sepia':
                result = sepia(img)
            else:
                result = img
            
            display_image(result, f"Result: {transformation}")
        
        with col2:
            st.subheader("Arithmetic Operations")
            operation = st.selectbox("Operation:", ["None", "Addition", "Subtraction", "Multiplication", "Division"])
            
            if operation != "None":
                value = st.slider(f"Value for {operation}:", 0, 255, 50)
                if operation == "Addition":
                    arith_result = cv2.add(img, value)
                elif operation == "Subtraction":
                    arith_result = cv2.subtract(img, value)
                elif operation == "Multiplication":
                    arith_result = cv2.multiply(img, value)
                elif operation == "Division":
                    arith_result = cv2.divide(img, max(value, 1))
                
                display_image(arith_result, f"Image after {operation} by {value}")

    # ==================== TAB 2: Color & Enhancement ====================
    with tab2:
        st.header("Color Manipulation & Image Enhancement")
        
        enhancement_type = st.radio("Enhancement Type:", ["Color Adjustment", "Brightness/Contrast", "Gamma Correction", "Sharpening"])
        
        if enhancement_type == "Color Adjustment":
            col1, col2, col3 = st.columns(3)
            with col1:
                r_scale = st.slider("Red Channel", 0.0, 3.0, 1.0, 0.05)
            with col2:
                g_scale = st.slider("Green Channel", 0.0, 3.0, 1.0, 0.05)
            with col3:
                b_scale = st.slider("Blue Channel", 0.0, 3.0, 1.0, 0.05)
            
            adjusted = img.copy().astype(np.float32)
            adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] * b_scale, 0, 255)
            adjusted[:, :, 1] = np.clip(adjusted[:, :, 1] * g_scale, 0, 255)
            adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] * r_scale, 0, 255)
            adjusted = adjusted.astype(np.uint8)
            show_comparison(img, adjusted, "Original", "Color Adjusted")
            
            # عمليات القنوات
            st.subheader("Channel Operations")
            channel_op = st.selectbox("Channel Operation:", ["None", "Swap R↔G", "Swap R↔B", "Swap G↔B", "Eliminate Red", "Eliminate Green", "Eliminate Blue"])
            if channel_op != "None":
                channel_result = img.copy()
                if channel_op == "Swap R↔G":
                    channel_result[:, :, [1, 2]] = channel_result[:, :, [2, 1]]
                elif channel_op == "Swap R↔B":
                    channel_result[:, :, [0, 2]] = channel_result[:, :, [2, 0]]
                elif channel_op == "Swap G↔B":
                    channel_result[:, :, [0, 1]] = channel_result[:, :, [1, 0]]
                elif channel_op == "Eliminate Red":
                    channel_result[:, :, 2] = 0
                elif channel_op == "Eliminate Green":
                    channel_result[:, :, 1] = 0
                elif channel_op == "Eliminate Blue":
                    channel_result[:, :, 0] = 0
                display_image(channel_result, f"After {channel_op}")
        
        elif enhancement_type == "Brightness/Contrast":
            brightness = st.slider("Brightness", -100, 100, 0)
            contrast = st.slider("Contrast", 0.5, 3.0, 1.0)
            enhanced = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
            show_comparison(img, enhanced, "Original", f"Brightness: {brightness}, Contrast: {contrast}")
        
        elif enhancement_type == "Gamma Correction":
            gamma = st.slider("Gamma Value", 0.1, 3.0, 1.0, 0.05)
            gamma_corrected = adjust_gamma(img, gamma)
            show_comparison(img, gamma_corrected, "Original", f"Gamma = {gamma}")
        
        elif enhancement_type == "Sharpening":
            sharpen_amount = st.slider("Sharpening Amount", 0.0, 3.0, 1.0, 0.1)
            sharpened = unsharp_mask(img, amount=sharpen_amount)
            show_comparison(img, sharpened, "Original", "Sharpened")
        
        # CLAHE Equalization
        st.subheader("Advanced Equalization")
        if st.button("Apply CLAHE Equalization"):
            clahe_result = clahe_equalization(img)
            show_comparison(img, clahe_result, "Original", "CLAHE Equalized")

    # ==================== TAB 3: Filters & Edges ====================
    with tab3:
        st.header("Filters and Edge Detection")
        
        filter_type = st.selectbox("Filter Type:", [
            "None", "Mean Filter", "Median Filter", "Gaussian Filter", 
            "Bilateral Filter", "Non-Local Means", "High Pass Filter", 
            "Laplacian Filter", "Sobel Edge", "Canny Edge", "Hough Lines"
        ])
        
        if filter_type != "None":
            if filter_type == "Mean Filter":
                kernel_size = st.slider("Kernel Size", 3, 15, 5, 2)
                filtered = cv2.blur(img, (kernel_size, kernel_size))
            elif filter_type == "Median Filter":
                kernel_size = st.slider("Kernel Size", 3, 15, 5, 2)
                filtered = cv2.medianBlur(img, kernel_size)
            elif filter_type == "Gaussian Filter":
                kernel_size = st.slider("Kernel Size", 3, 15, 5, 2)
                sigma = st.slider("Sigma", 0.1, 5.0, 1.0)
                filtered = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            elif filter_type == "Bilateral Filter":
                d = st.slider("Diameter", 3, 15, 9)
                sigma_color = st.slider("Sigma Color", 10, 150, 75)
                sigma_space = st.slider("Sigma Space", 10, 150, 75)
                filtered = apply_bilateral_filter(img, d, sigma_color, sigma_space)
            elif filter_type == "Non-Local Means":
                h = st.slider("Filter Strength", 3, 15, 10)
                filtered = apply_non_local_means(img, h)
            elif filter_type == "High Pass Filter":
                filtered = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
                filtered = np.uint8(np.absolute(filtered))
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            elif filter_type == "Laplacian Filter":
                filtered = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
                filtered = np.uint8(np.absolute(filtered))
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            elif filter_type == "Sobel Edge":
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                filtered = np.sqrt(sobelx**2 + sobely**2)
                filtered = np.uint8(np.clip(filtered, 0, 255))
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            elif filter_type == "Canny Edge":
                low = st.slider("Low Threshold", 0, 200, 50)
                high = st.slider("High Threshold", 0, 300, 150)
                filtered = apply_canny_edge(img, low, high)
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            elif filter_type == "Hough Lines":
                filtered = apply_hough_lines(img)
            
            show_comparison(img, filtered, "Original", filter_type)

    # ==================== TAB 4: Morphology & Segmentation ====================
    with tab4:
        st.header("Mathematical Morphology & Image Segmentation")
        
        morph_type = st.selectbox("Morphology Operation:", [
            "None", "Erosion", "Dilation", "Opening", "Closing", 
            "Gradient", "Top Hat", "Black Hat", "Internal Boundary", 
            "External Boundary", "Watershed Segmentation"
        ])
        
        if morph_type != "None":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            
            if morph_type == "Erosion":
                result = cv2.erode(binary, kernel, iterations=1)
            elif morph_type == "Dilation":
                result = cv2.dilate(binary, kernel, iterations=1)
            elif morph_type == "Opening":
                result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            elif morph_type == "Closing":
                result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            elif morph_type == "Gradient":
                result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
            elif morph_type == "Top Hat":
                result = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
            elif morph_type == "Black Hat":
                result = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
            elif morph_type == "Internal Boundary":
                eroded = cv2.erode(binary, kernel)
                result = cv2.subtract(binary, eroded)
            elif morph_type == "External Boundary":
                dilated = cv2.dilate(binary, kernel)
                result = cv2.subtract(dilated, binary)
            elif morph_type == "Watershed Segmentation":
                result = apply_watershed_segmentation(img)
                display_image(result, "Watershed Segmentation Result")
                result = None
            
            if result is not None:
                st.subheader("Original vs Processed")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(binary, caption="Binary Image", use_container_width=True, channels="GRAY")
                with col2:
                    st.image(result, caption=f"{morph_type}", use_container_width=True, channels="GRAY")
        
        # Corner and Blob Detection
        st.subheader("Feature Detection")
        feature_type = st.selectbox("Feature Type:", ["None", "Corner Detection", "Blob Detection"])
        if feature_type == "Corner Detection":
            corners_img = detect_corners(img)
            show_comparison(img, corners_img, "Original", "Corners Detected")
        elif feature_type == "Blob Detection":
            blobs_img = detect_blobs(img)
            show_comparison(img, blobs_img, "Original", "Blobs Detected")

    # ==================== TAB 5: Histogram & Threshold ====================
    with tab5:
        st.header("Histogram Analysis & Thresholding")
        
        # عرض الرسم البياني
        if st.checkbox("Show Histogram"):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            colors = ['blue', 'green', 'red']
            titles = ['Blue Channel', 'Green Channel', 'Red Channel']
            for i, (ax, color, title) in enumerate(zip(axes, colors, titles)):
                ax.hist(img[:, :, i].ravel(), bins=256, color=color, alpha=0.7)
                ax.set_title(title)
                ax.set_xlim([0, 256])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # تحسين الرسم البياني
        st.subheader("Histogram Enhancement")
        hist_op = st.selectbox("Enhancement Method:", ["None", "Histogram Equalization", "CLAHE", "Histogram Stretching"])
        
        if hist_op == "Histogram Equalization":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            equalized = cv2.equalizeHist(gray)
            equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            show_comparison(img, equalized, "Original", "Histogram Equalized")
        elif hist_op == "CLAHE":
            clahe_result = clahe_equalization(img)
            show_comparison(img, clahe_result, "Original", "CLAHE")
        elif hist_op == "Histogram Stretching":
            stretched = np.zeros_like(img)
            for i in range(3):
                stretched[:, :, i] = np.uint8(255 * (img[:, :, i] - img[:, :, i].min()) / 
                                              (img[:, :, i].max() - img[:, :, i].min() + 1e-7))
            show_comparison(img, stretched, "Original", "Histogram Stretched")
        
        # التحتيم
        st.subheader("Thresholding Methods")
        thresh_type = st.selectbox("Threshold Method:", ["None", "Global Threshold", "Adaptive Mean", "Adaptive Gaussian", "Otsu"])
        
        if thresh_type != "None":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if thresh_type == "Global Threshold":
                thresh_val = st.slider("Threshold Value", 0, 255, 127)
                _, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            elif thresh_type == "Adaptive Mean":
                block_size = st.slider("Block Size", 3, 31, 11, 2)
                c = st.slider("Constant C", 0, 20, 2)
                thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
            elif thresh_type == "Adaptive Gaussian":
                block_size = st.slider("Block Size", 3, 31, 11, 2)
                c = st.slider("Constant C", 0, 20, 2)
                thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
            elif thresh_type == "Otsu":
                _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            thresh_rgb = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)
            show_comparison(img, thresh_rgb, "Original", f"{thresh_type} Thresholding")

    # ==================== TAB 6: Artistic Effects ====================
    with tab6:
        st.header("Artistic Effects")
        
        effect = st.selectbox("Choose Effect:", [
            "None", "Pencil Sketch", "Cartoonize", 
            "Vignette", "Sepia Tone", "Invert Colors", "Emboss"
        ])
        
        if effect == "Pencil Sketch":
            result = pencil_sketch(img)
            show_comparison(img, result, "Original", "Pencil Sketch")
        elif effect == "Cartoonize":
            result = cartoonize(img)
            show_comparison(img, result, "Original", "Cartoon Effect")
        elif effect == "Vignette":
            intensity = st.slider("Vignette Intensity", 0.1, 2.0, 0.5)
            result = vignette(img, intensity)
            show_comparison(img, result, "Original", "Vignette Effect")
        elif effect == "Sepia Tone":
            result = sepia(img)
            show_comparison(img, result, "Original", "Sepia Tone")
        elif effect == "Invert Colors":
            result = invert_colors(img)
            show_comparison(img, result, "Original", "Inverted Colors")
        elif effect == "Emboss":
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            emboss = cv2.filter2D(gray, -1, kernel) + 128
            result = cv2.cvtColor(emboss, cv2.COLOR_GRAY2RGB)
            show_comparison(img, result, "Original", "Emboss Effect")

    # ==================== TAB 7: Frequency Domain ====================
    with tab7:
        st.header("Frequency Domain Processing (Fourier Transform)")
        
        st.info("The Fourier Transform decomposes an image into its sine and cosine components.")
        
        if st.button("Show Fourier Transform"):
            magnitude = apply_fourier_transform(img)
            st.image(magnitude, caption="Frequency Spectrum", use_container_width=True, channels="GRAY")
        
        filter_freq = st.selectbox("Frequency Filter:", ["None", "Low Pass Filter", "High Pass Filter"])
        
        if filter_freq != "None":
            cutoff = st.slider("Cutoff Frequency (Radius)", 10, 100, 30)
            if filter_freq == "Low Pass Filter":
                filtered = apply_low_pass_filter_fourier(img, cutoff)
            else:
                filtered = apply_high_pass_filter_fourier(img, cutoff)
            
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
            show_comparison(img, filtered_rgb, "Original", f"{filter_freq} (cutoff={cutoff})")

    # ==================== TAB 8: Advanced Features ====================
    with tab8:
        st.header("Advanced Features")
        
        # إضافة الضوضاء وإزالتها
        st.subheader("Noise Addition & Removal")
        
        noise_type = st.selectbox("Add Noise:", ["None", "Salt & Pepper", "Gaussian", "Speckle"])
        
        if noise_type != "None":
            noisy = img.copy()
            if noise_type == "Salt & Pepper":
                salt_prob = st.slider("Noise Amount", 0.01, 0.2, 0.05)
                noisy = img.copy()
                num_salt = int(salt_prob * img.size * 0.5)
                coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
                noisy[coords[0], coords[1]] = 255
                num_pepper = int(salt_prob * img.size * 0.5)
                coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
                noisy[coords[0], coords[1]] = 0
            elif noise_type == "Gaussian":
                sigma = st.slider("Sigma", 10, 100, 25)
                gauss = np.random.normal(0, sigma, img.shape)
                noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)
            elif noise_type == "Speckle":
                gauss = np.random.normal(0, 0.1, img.shape)
                noisy = np.clip(img + img * gauss, 0, 255).astype(np.uint8)
            
            display_image(noisy, f"Noisy Image ({noise_type})")
            
            # إزالة الضوضاء
            denoise_type = st.selectbox("Denoise Filter:", ["None", "Median Filter", "Gaussian Filter", "Bilateral Filter"])
            
            if denoise_type != "None":
                if denoise_type == "Median Filter":
                    kernel = st.slider("Kernel Size", 3, 15, 5, 2)
                    denoised = cv2.medianBlur(noisy, kernel)
                elif denoise_type == "Gaussian Filter":
                    kernel = st.slider("Kernel Size", 3, 15, 5, 2)
                    denoised = cv2.GaussianBlur(noisy, (kernel, kernel), 0)
                elif denoise_type == "Bilateral Filter":
                    denoised = cv2.bilateralFilter(noisy, 9, 75, 75)
                
                show_comparison(noisy, denoised, "Noisy Image", f"After {denoise_type}")
        
        # Image Blending
        st.subheader("Image Blending")
        if st.checkbox("Enable Image Blending"):
            st.info("Upload a second image to blend with the original")
            second_file = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"], key="blend")
            if second_file is not None:
                img2 = load_image(second_file)
                img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
                alpha = st.slider("Alpha (Weight of first image)", 0.0, 1.0, 0.5)
                blended = cv2.addWeighted(img, alpha, img2, 1 - alpha, 0)
                show_comparison(img, blended, "First Image", f"Blended (alpha={alpha})")

else:
    # عرض عند عدم رفع صورة
    st.info("👈 Please upload an image from the sidebar to start processing!")
    
    st.markdown("""
    ### ✅ Features Available:
    
    **🎯 Basic Operations**
    - Grayscale, Solarization, Complement, Invert, Sepia
    - Arithmetic operations (Add, Subtract, Multiply, Divide)
    
    **🌈 Color & Enhancement**
    - RGB channel adjustment
    - Brightness, Contrast, Gamma correction
    - Advanced sharpening
    
    **🔍 Filters & Edge Detection**
    - Multiple blur filters (Mean, Median, Gaussian, Bilateral)
    - Edge detection (Sobel, Canny, Laplacian)
    - Line detection (Hough Transform)
    
    **🧬 Morphology & Segmentation**
    - Erosion, Dilation, Opening, Closing
    - Watershed segmentation
    - Corner and blob detection
    
    **📊 Histogram & Thresholding**
    - Histogram visualization
    - Equalization, CLAHE, stretching
    - Global, Adaptive, and Otsu thresholding
    
    **🎨 Artistic Effects**
    - Pencil sketch, Cartoonize
    - Vignette, Sepia, Emboss
    
    **⚡ Frequency Domain**
    - Fourier Transform visualization
    - Low/High pass filters
    
    **🤖 Advanced Features**
    - Noise addition and removal
    - Image blending
    """)