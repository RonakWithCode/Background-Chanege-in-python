from pixellib.tune_bg import alter_bg

bg = alter_bg()

bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

bg.change_bg_img(f_image_path = "mainImage.jpg",b_image_path = "background.jpg", output_image_name="image.jpg")