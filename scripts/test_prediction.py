from scripts.predict import predict
from loguru import logger

# iPhone 13 specifications
iphone_13_specs = {
    'Battery': 3240,  # mAh
    'Ram': 6144,      # 6GB in MB
    'Display': 6.1,   # inches
    'Camera': 24,     # Dual 12MP cameras
    'External_Memory': False,  # No external memory support
    'Inbuilt_memory': 128,    # GB
    'fast_charging': 20,      # W
    'Screen_resolution': "1170x2532",
    'No_of_sim': ["Dual Sim", "5G", "VoLTE"],
    'Android_version': "15.0",  # iOS equivalent
    'company': "Apple",
    'Processor': "Apple",
    'Processor_name': "A15 Bionic"
}

def main():
    try:
        print("\nTesting prediction for iPhone 13:")
        print("\nSpecifications:")
        for key, value in iphone_13_specs.items():
            print(f"{key}: {value}")
            
        prediction = predict(iphone_13_specs)
        print(f"\nPredicted Price: ₹{prediction[0]:,.2f}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    main() 