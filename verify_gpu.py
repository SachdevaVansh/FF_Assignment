import torch
import os
from PIL import Image
from datautils import batch_remove_background_and_make_bw, resize_rotate_and_pad_gpu, resize_rotate_and_pad

def test_gpu_pipeline():
    print("="*50)
    print("Running GPU Verification Script")
    print("="*50)
    
    # 1. Check CUDA Availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available. The code will fallback to CPU.")

    # 2. Test Image Creation
    img_name = "test_gpu_image.png"
    print(f"\nCreating dummy image: {img_name}")
    Image.new('RGB', (500, 500), color='blue').save(img_name)

    # 3. specific function testing
    try:
        if cuda_available:
            print("\nTesting resize_rotate_and_pad_gpu (Direct Tensor Call)...")
            from torchvision.transforms import v2
            with Image.open(img_name) as img:
                tensor_img = v2.functional.to_image(img).to("cuda")
                out_tensor = resize_rotate_and_pad_gpu(tensor_img, final_size=128)
                print(f"Output Tensor Shape: {out_tensor.shape} (Expected: [3, 128, 128])")
                assert out_tensor.shape == (3, 128, 128)
                print("Direct GPU function test PASSED.")
    except Exception as e:
        print(f"Direct GPU function test FAILED: {e}")

    # 4. Full Pipeline Test
    print(f"\nTesting full pipeline 'batch_remove_background_and_make_bw'...")
    try:
        batch_remove_background_and_make_bw([img_name], use_gpu=True, rotate=True)
        
        if os.path.exists(img_name):
            with Image.open(img_name) as img:
                print(f"Pipeline Result Image Size: {img.size}")
                print(f"Pipeline Result Image Mode: {img.mode}")
                if img.size == (128, 128):
                    print("Pipeline verification PASSED.")
                else:
                    print("Pipeline verification FAILED: Incorrect size.")
        else:
            print("Pipeline verification FAILED: Output file not found.")
            
    except Exception as e:
        print(f"Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    if os.path.exists(img_name):
        os.remove(img_name)
    print("\nVerification complete.")

if __name__ == "__main__":
    test_gpu_pipeline()
