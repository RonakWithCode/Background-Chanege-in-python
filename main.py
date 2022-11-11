from pixellib.tune_bg import alter_bg

bg = alter_bg()
# To download go on : https://objects.githubusercontent.com/github-production-release-asset-2e65be/255074156/53c11380-90ee-11ea-905d-412859743640?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20221111%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20221111T145942Z&X-Amz-Expires=300&X-Amz-Signature=a6353a00282fc5b83fed463e493dd62befb096a23e1d98d1cf9c6cbeed9e07fe&X-Amz-SignedHeaders=host&actor_id=98526174&key_id=0&repo_id=255074156&response-content-disposition=attachment%3B%20filename%3Ddeeplabv3_xception_tf_dim_ordering_tf_kernels.h5&response-content-type=application%2Foctet-stream

bg.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")

bg.change_bg_img(f_image_path = "mainImage.jpg",b_image_path = "background.jpg", output_image_name="image.jpg")
