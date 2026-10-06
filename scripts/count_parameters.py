"""Count actual model parameters without publishing planning estimates."""
import sys
from .profile import main

if __name__ == '__main__':
    sys.argv.append('--parameters-only')
    main()
