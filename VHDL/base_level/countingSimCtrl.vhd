LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY countingSimCtrl IS
	GENERIC (n : INTEGER := 10 ); --- bit pointer to memory
	PORT (
		clk, rst 				: IN STD_LOGIC;
		run		 				: IN STD_LOGIC;
		runOut, done 			: OUT STD_LOGIC;
		pointer 				: OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0) --- As of now only support up to 16 classes so 4'bits 
	);
END ENTITY countingSimCtrl;

ARCHITECTURE ctrl OF countingSimCtrl IS
	COMPONENT popCount IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk , rst 	: IN STD_LOGIC;
			en		 	: IN STD_LOGIC;
			dout        : OUT  STD_LOGIC_VECTOR (lenPop-1 DOWNTO 0)
		);
	END COMPONENT;
	SIGNAL count : STD_LOGIC_VECTOR (n-1 DOWNTO 0);
	TYPE state IS  (init,  counting, updatewaite);
	SIGNAL ns,  ps : state;
	SIGNAL countEn , countRst : STD_LOGIC;
BEGIN

	PROCESS(clk) BEGIN 
		IF rising_edge(clk) then
			IF (rst ='1')then
				ps <= init; 
			ELSE  
				ps <= ns;  
			END IF;
		END IF;
	END PROCESS;
	
	PROCESS ( ps,  run, count)
	BEGIN 
	runOut <= '0';
	countRst <= '0';
	countEn <= '0';	
    done <= '0';
		CASE (ps) IS 
			WHEN init =>
				countRst <= '1';
				IF ( run = '1') THEN
					ns <= updatewaite;
				ELSE
					ns <= init;
				END IF;
			WHEN updatewaite =>
				ns <= counting;
			WHEN counting =>
				IF (count = STD_LOGIC_VECTOR(TO_UNSIGNED((2**n)-1 , n))) THEN  ---
					done <= '1';
					IF ( run = '1') THEN
						countRst <= '1';
						ns <= counting;
					ELSE
						ns <= init;
					END IF;
				ELSE 
					runOut<= '1';
					countEn <= '1';	
					ns <= counting;
				END IF;
			WHEN OTHERS =>
					ns <= init;
		END CASE;
	END PROCESS;
	
	sel : popCount 
		GENERIC MAP(n)
		PORT MAP(
			clk, countRst, countEn, count
		);
	pointer <= count;

END ARCHITECTURE;	
