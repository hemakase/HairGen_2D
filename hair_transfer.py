import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import sys
import gdown

# 현재 디렉토리를 파이썬 패스에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from networks import get_network
import torchvision.transforms as std_trnsf

# 모델 체크포인트 다운로드 함수
def download_model_weights(save_path):
    if not os.path.exists(save_path):
        print("Downloading model weights...")
        url = 'https://drive.google.com/uc?id=1w7oMuxckqEClImjLFTH7xBCpm1wg7Eg4'
        gdown.download(url, save_path, quiet=False)
        print(f"Model weights downloaded to {save_path}")
    else:
        print(f"Model weights already exist at {save_path}")

# 머리카락 세그멘테이션 함수
def segment_hair(net, img_path, device):
    # 이미지 변환 정의
    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 이미지 로드 및 변환
    img = Image.open(img_path).convert('RGB')  # RGB로 변환
    data = test_image_transforms(img)
    data = torch.unsqueeze(data, dim=0)
    net.eval()
    data = data.to(device)
    
    # 추론
    with torch.no_grad():
        logit = net(data)
    
    # 마스크 생성
    pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()
    mask = pred >= 0.5
    
    # 원본 이미지
    img_np = np.array(img)
    
    # 마스크 오버레이 생성
    mh, mw = data.size(2), data.size(3)
    mask_overlay = np.zeros((mh, mw, 3), dtype=np.uint8)
    mask_overlay[:,:,0] = 255  # 빨간색 마스크
    mask_overlay = mask_overlay * mask[:,:,np.newaxis]
    
    # 원본 이미지와 동일한 크기로 마스크 조정
    ih, iw, _ = img_np.shape
    delta_h = mh - ih
    delta_w = mw - iw
    
    top = delta_h // 2
    bottom = mh - (delta_h - top)
    left = delta_w // 2
    right = mw - (delta_w - left)
    
    mask_overlay = mask_overlay[top:bottom, left:right, :]
    
    return img_np, mask_overlay

def transfer_hair(source_img, source_mask, target_img, target_mask=None, scale_factor=1.2):
    """
    소스 이미지의 머리카락을 타겟 이미지에 덧씌우는 함수
    :param scale_factor: 머리카락 확대 비율 (기본값 1.5배)
    """
    # 이미지 크기 확인 및 조정
    if source_img.shape != target_img.shape or source_mask.shape[:2] != target_img.shape[:2]:
        source_mask = cv2.resize(source_mask, (target_img.shape[1], target_img.shape[0]))
        source_img = cv2.resize(source_img, (target_img.shape[1], target_img.shape[0]))
    
    # 머리카락 부분만 마스크로 추출 (빨간색 채널이 있는 부분)
    source_hair_mask = (source_mask[:,:,0] > 0).astype(np.uint8)
    
    # 1.5배 확대 적용
    if scale_factor != 1.0:
        # 마스크 중심점 계산
        source_moments = cv2.moments(source_hair_mask)
        if source_moments["m00"] != 0:
            source_cx = int(source_moments["m10"] / source_moments["m00"])
            source_cy = int(source_moments["m01"] / source_moments["m00"])
        else:
            source_cx = source_hair_mask.shape[1] // 2
            source_cy = source_hair_mask.shape[0] // 2

        # 확대 변환 행렬 생성
        M = cv2.getRotationMatrix2D((source_cx, source_cy), 0, scale_factor)
        
        # 아핀 변환 적용
        source_hair_mask = cv2.warpAffine(
            source_hair_mask, M, 
            (source_hair_mask.shape[1], source_hair_mask.shape[0]),
            flags=cv2.INTER_LINEAR
        )
        source_img = cv2.warpAffine(
            source_img, M, 
            (source_img.shape[1], source_img.shape[0]),
            flags=cv2.INTER_LINEAR
        )

    # 타겟 이미지의 머리카락 마스크 추출 또는 생성
    if target_mask is not None:
        target_hair_mask = (target_mask[:,:,0] > 0).astype(np.uint8)
    else:
        target_hair_mask = np.zeros_like(source_hair_mask)

    # 중심점 재정렬
    source_moments = cv2.moments(source_hair_mask)
    if source_moments["m00"] != 0:
        source_cx = int(source_moments["m10"] / source_moments["m00"])
        source_cy = int(source_moments["m01"] / source_moments["m00"])
    else:
        source_cx = source_hair_mask.shape[1] // 2
        source_cy = source_hair_mask.shape[0] // 2

    target_moments = cv2.moments(target_hair_mask)
    if target_moments["m00"] != 0:
        target_cx = int(target_moments["m10"] / target_moments["m00"])
        target_cy = int(target_moments["m01"] / target_moments["m00"])
    else:
        target_cx = target_img.shape[1] // 2
        target_cy = target_img.shape[0] // 3  # 얼굴 위치 고려

    # 이동 거리 계산 및 변환 적용
    dx = target_cx - source_cx
    dy = target_cy - source_cy + 10  # 추가 오프셋 적용 - 10픽셀 더 아래로 이동
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned_source_mask = cv2.warpAffine(source_hair_mask, M, (source_hair_mask.shape[1], source_hair_mask.shape[0]))
    aligned_source_img = cv2.warpAffine(source_img, M, (source_img.shape[1], source_img.shape[0]))

    # 타겟의 머리카락 부분을 피부색으로 대체
    if target_mask is not None:
        # 타겟 머리카락 마스크 확장 (dilate)하여 더 넓은 영역 커버
        kernel = np.ones((7, 7), np.uint8)
        dilated_target_mask = cv2.dilate(target_hair_mask, kernel, iterations=2)
        
        # 타겟 머리카락 마스크에 블러 적용 (더 강한 블러)
        blurred_target_mask = cv2.GaussianBlur(dilated_target_mask.astype(np.float32), (25, 25), 0)
        blurred_target_mask_3channel = np.stack([blurred_target_mask, blurred_target_mask, blurred_target_mask], axis=2)
        
        # 단순화된 피부색 추출 - 얼굴 아래쪽에서 고정된 위치의 피부색 샘플링
        face_y = min(target_img.shape[0]-100, target_cy + 70)  # 머리카락 아래쪽으로
        face_x = target_cx
        
        # 얼굴 피부색 샘플 추출 (단일 영역)
        sample_size = 30
        y_min = max(0, face_y - sample_size//2)
        y_max = min(target_img.shape[0], face_y + sample_size//2)
        x_min = max(0, face_x - sample_size//2)
        x_max = min(target_img.shape[1], face_x + sample_size//2)
        
        # 샘플 영역에서 평균 피부색 계산
        skin_sample = target_img[y_min:y_max, x_min:x_max].copy()
        avg_skin_color = np.mean(skin_sample.reshape(-1, 3), axis=0)
                
        # 피부색으로 머리카락 영역 채우기 (블렌딩 강화)
        skin_color_fill = np.ones_like(target_img) * avg_skin_color.reshape(1, 1, 3)
        target_img_no_hair = target_img * (1 - blurred_target_mask_3channel) + skin_color_fill * blurred_target_mask_3channel
    else:
        target_img_no_hair = target_img.copy()

    # 경계 부드럽게 처리 (가우시안 블러 적용)
    hair_mask = cv2.GaussianBlur(aligned_source_mask.astype(np.float32), (5,5), 0)
    hair_mask_3channel = np.stack([hair_mask, hair_mask, hair_mask], axis=2)

    # 타겟 이미지 합성 (원래 머리카락이 지워진 이미지에 새 머리카락 합성)
    target_without_hair = target_img_no_hair * (1 - hair_mask_3channel)
    source_hair = aligned_source_img * hair_mask_3channel
    result = target_without_hair + source_hair

    return result.astype(np.uint8)

if __name__ == "__main__":
    # 결과 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 모델 체크포인트 다운로드
    model_path = './models/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth'
    download_model_weights(model_path)
    
    # 기기 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 네트워크 로드
    net = get_network('pspnet_resnet101').to(device)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state['weight'])
    
    # 이미지 경로 설정
    dongchan_path = r'D:\Coding\HairGen_2D\images\dongchan.png'
    unu_path = r'D:\Coding\HairGen_2D\images\unu.png'
    
    print(f"이미지 로드 및 세그멘테이션 중: {dongchan_path}, {unu_path}")
    
    # 세그멘테이션 실행
    dongchan_img, dongchan_mask = segment_hair(net, dongchan_path, device)
    unu_img, unu_mask = segment_hair(net, unu_path, device)
    
    # 머리카락 전이 실행
    print("unu의 머리카락을 dongchan에게 덧씌우는 중...")
    print("머리카락 중심점을 정렬하여 전이 수행...")
    transfer_result = transfer_hair(unu_img, unu_mask, dongchan_img, dongchan_mask)
    
    # 결과 저장 (테두리와 제목 없이 이미지만 저장)
    output_path = os.path.join('results', 'hair_transfer_result.png')
    cv2.imwrite(output_path, cv2.cvtColor(transfer_result, cv2.COLOR_RGB2BGR))
    print(f"머리카락 전이 결과가 {output_path}에 저장되었습니다.") 