LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY SeqAdderCtrl IS
	GENERIC (ceilingLogPop : INTEGER := 3;   -- ceilingLOG2(#popCounters)
			nPop : INTEGER := 8 );			-- #popCounters
	PORT (
		clk, rst 				: IN STD_LOGIC;
		run		 				: IN STD_LOGIC;
		reg1Update, reg1rst 	: OUT STD_LOGIC;
		reg2Update, reg2rst 	: OUT STD_LOGIC;
		muxSel 					: OUT STD_LOGIC_VECTOR(ceilingLogPop DOWNTO 0)
	);
END ENTITY SeqAdderCtrl;

ARCHITECTURE ctrl OF SeqAdderCtrl IS
	COMPONENT popCount IS
		GENERIC (lenPop : INTEGER := 8);   -- bit width out popCounters
		PORT (
			clk , rst 	: IN STD_LOGIC;
			en		 	: IN STD_LOGIC;
			dout        : OUT  STD_LOGIC_VECTOR (lenPop-1 DOWNTO 0)
		);
	END COMPONENT;
	SIGNAL count : STD_LOGIC_VECTOR (ceilingLogPop DOWNTO 0);
	TYPE state IS  (init,  sum);
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
	countEn <= '0';
    countRst <= '0';
	reg1Update <= '0';
    reg1rst <= '0';
	reg2Update <= '0';
		CASE (ps) IS 
			WHEN init =>
					countRst <= '1';
					reg1rst <= '1';
				IF ( run = '1') then
					ns <= sum;
				ELSE
					ns <= init;
				END IF;
			WHEN sum =>
				
				IF ( count = STD_LOGIC_VECTOR(to_UNSIGNED(nPop , count'length))) then  --- perhaps -1 is extra! check
					countRst <= '1';
					IF ( run = '1') then
						ns <= sum;
						reg1rst <= '1';
						reg2Update <= '1';
					ELSE
						ns <= init;
						reg2Update <= '1';
					END IF;
				ELSE 
					countEn <= '1';	
					reg1Update <= '1';
					ns <= sum;
				END IF;
			WHEN OTHERS =>
					ns <= init;
		END CASE;
	END PROCESS;
	
	reg2rst <= rst;
	
	sel : popCount 
		GENERIC MAP(ceilingLogPop +1)
		PORT MAP(
			clk , countRst 	,countEn	, count
		);
	muxSel <= count;
	
END ARCHITECTURE ctrl;