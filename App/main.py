import sys
import os
import json
from backend.abac_pipeline.pipeline import ABACPipeline

def main():
    print("=====================================================")
    print("CHƯƠNG TRÌNH TRÍCH XUẤT CHÍNH SÁCH ABAC TỪ VĂN BẢN")
    print("=====================================================")
    
    policy_name = input("Nhập tên Policy (dùng để đặt tên file lưu kết quả): ").strip()
    if not policy_name:
        policy_name = "default_policy"
        
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, f"{policy_name}.json")
    
    saved_results = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                saved_results = json.load(f)
                print(f"Đã tìm thấy file '{file_path}', sẽ tiếp tục lưu nối tiếp.")
        except json.JSONDecodeError:
            print(f"File '{file_path}' bị lỗi định dạng, sẽ tạo mới.")
            saved_results = []
            
    id_counter = len(saved_results) + 1

    print("\nĐang tải mô hình ABAC Pipeline, vui lòng đợi...")
    try:
        pipeline = ABACPipeline()
        print("Tải mô hình thành công!\n")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        sys.exit(1)
        
    print("Nhập 'exit', 'quit' hoặc 'done' để thoát.")
    print("-----------------------------------------------------")
    
    while True:
        try:
            policy_text = input(f"\n[{id_counter}] Nhập câu ACP (tiếng Anh): ").strip()
            if policy_text.lower() in ['exit', 'quit', 'done']:
                print("Đang thoát chương trình...")
                break
            if not policy_text:
                continue
                
            print("Đang xử lý...")
            result = pipeline.process_policy(policy_text)
            
            # In kết quả ra màn hình
            print("\n--- Kết quả Trích xuất ---")
            print(f"Effect (Hiệu lực): {result.rule.effect}")
            
            print("Subjects (Chủ thể):")
            if not result.rule.subjects:
                print("  (Không có)")
            for s in result.rule.subjects:
                print(f"  - Attribute: {s.attribute}, Operator: {s.operator}, Values: {s.values}")
                
            print(f"Actions (Hành động): {result.rule.actions}")
            
            print("Resources (Tài nguyên):")
            if not result.rule.resources:
                print("  (Không có)")
            for r in result.rule.resources:
                print(f"  - Attribute: {r.attribute}, Operator: {r.operator}, Values: {r.values}")
                
            print("Constraints (Ràng buộc):")
            if not result.rule.constraints:
                print("  (Không có)")
            for c in result.rule.constraints:
                print(f"  - Attribute: {c.attribute}, Operator: {c.operator}, Values: {c.values}, Ref: {c.attribute_ref}")
                
            # Lưu kết quả vào dict
            record = {
                "id": id_counter,
                "policy_text": policy_text,
                "rule": {
                    "effect": result.rule.effect,
                    "subjects": [{"attribute": s.attribute, "operator": s.operator, "values": s.values} for s in result.rule.subjects],
                    "actions": result.rule.actions,
                    "resources": [{"attribute": r.attribute, "operator": r.operator, "values": r.values} for r in result.rule.resources],
                    "constraints": [{"attribute": c.attribute, "operator": c.operator, "values": c.values, "attribute_ref": c.attribute_ref} for c in result.rule.constraints]
                }
            }
            
            saved_results.append(record)
            
            # Ghi vào file json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(saved_results, f, ensure_ascii=False, indent=4)
                
            print(f"\n[+] Đã lưu kết quả (ID: {id_counter}) vào file {file_path}")
            print("--------------------------\n")
            
            id_counter += 1
            
        except KeyboardInterrupt:
            print("\nĐã ngắt bằng phím tắt. Đang thoát...")
            break
        except Exception as e:
            print(f"\nCó lỗi xảy ra trong quá trình xử lý: {e}")

if __name__ == "__main__":
    main()
